"""P666 多种子蒙特卡洛评测：官方环境直跑（不经容器），双口径输出。

设计要点（避免"第二套评分实现"）:
- 官方配置只读：load_suite(configs/public-suite-v1.yaml)；uniform 取 groups[0].cases[0].task_config，
  crossing 取 groups[1].cases[0].task_config。
- 策略载入：把仓库根与提交目录插入 sys.path 后 **静态 import entry**（不做任何动态加载）。
- 每局：env.reset(seed) -> 每台机器人各建一个策略实例 + reset(EpisodeContext) -> 循环 act/step。
- 奖励与指标一律取 infos[agent]["reward_terms"] / infos[agent]["metrics"]（官方 metrics/rewards），
  本脚本不重算 C_t / K_t。
- 口径：私有种子（默认 50 例/组）与公开 4 例（官方 suite 原样，policy_repeats=2）同时给出。
- 分数：S = 500 * (J_bar_basic + J_bar_coop)；CI95 用 bootstrap 重采样 scenario 均值
  （1000 次、seed=81273、2.5/97.5 分位）。
- 退出码：所有局 status == "ok" -> 0；出现任何非 ok 局 -> 3。

用法:
    .venv\\Scripts\\python.exe participant\\P666\\tools_local\\eval_mc.py ^
        --output outputs\\P666\\E001 --tag baseline
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径与静态导入
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = REPO_ROOT / "participant" / "P666"
TOOLS_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = REPO_ROOT / "participant" / "P666" / "artifacts"
SUITE_PATH = REPO_ROOT / "configs" / "public-suite-v1.yaml"

for _p in (str(TOOLS_DIR), str(SUBMISSION_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.protocol import (  # noqa: E402
    BuildContext,
    EpisodeContext,
    PublicTaskParams,
    ResourceLimits,
    get_protocol_spec,
)
from coverage_bench.suites import EvaluationSuite, load_suite  # noqa: E402

import entry  # noqa: E402  # 参赛策略（静态导入）
import seedset  # noqa: E402

# ---------------------------------------------------------------------------
# 常量（bootstrap 参数与官方 default_scoring_config 一致）
# ---------------------------------------------------------------------------

SCORE_SCALE = 500.0          # S = 500 * (J_bar_basic + J_bar_coop)
BOOTSTRAP_RESAMPLES = 1000   # 官方 bootstrap_resamples
BOOTSTRAP_SEED = 81273       # 官方 bootstrap_seed
CONFIDENCE_LEVEL = 0.95      # 官方 confidence_level
GROUP_UNIFORM_INDEX = 0      # groups[0] = basic (uniform)
GROUP_CROSSING_INDEX = 1     # groups[1] = cooperation (crossing)


class ProtocolViolation(RuntimeError):
    """动作不符合 coverage-policy/1.0（官方不裁剪非法动作，本脚本自行拦截并记录）。"""


def _force_utf8_stdout() -> None:
    """Windows 控制台默认代码页会让中文汇总表变成乱码；这里显式切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 输出流可能已被重定向且不可重配置
            pass


# ---------------------------------------------------------------------------
# 官方等价的小工具
# ---------------------------------------------------------------------------


def _public_task_params(task_config: Any) -> PublicTaskParams:
    """把 TaskConfig.public 映射成协议的 PublicTaskParams（与 runtime.py 同构的只读映射）。"""
    task = task_config.public
    return PublicTaskParams(
        map_half_extent=task.map_half_extent,
        dt=task.dt,
        robot_radius=task.robot_radius,
        robot_mass=task.robot_mass,
        drive_force=task.drive_force,
        damping=task.damping,
        robot_max_speed=task.robot_max_speed,
        contact_force=task.contact_force,
        contact_margin=task.contact_margin,
        target_radius=task.target_radius,
        target_max_speed=task.target_max_speed,
        sense_radius=task.sense_radius,
        motion_kind=task.motion_kind,
        turn_interval_steps=tuple(task.turn_interval_steps),
        target_speed_fraction=tuple(task.target_speed_fraction),
        robot_boundary=task.robot_boundary,
        target_boundary=task.target_boundary,
    )


def _check_action(action: Any) -> Optional[str]:
    """返回 None 表示动作合法；否则返回问题描述（口径同 validation.validate_action）。"""
    if not isinstance(action, np.ndarray):
        return f"动作不是 ndarray: {type(action)}"
    if action.dtype != np.float32:
        return f"dtype 不是 float32: {action.dtype}"
    if action.shape != (2,):
        return f"shape 不是 (2,): {action.shape}"
    if not np.isfinite(action).all():
        return "动作含非有限值"
    if np.any(action < -1.0) or np.any(action > 1.0):
        return f"动作越界 [-1,1]: {action.tolist()}"
    return None


def _policy_seed(scenario_seed: int, repeat_index: int, agent_index: int) -> int:
    """本工具的确定性 policy_seed 派生（官方用随机调度表；规则策略不依赖它）。"""
    return int((scenario_seed * 1000003 + repeat_index * 10007 + agent_index) % (2**63))


# ---------------------------------------------------------------------------
# 策略工厂
# ---------------------------------------------------------------------------

PolicyFactory = Callable[[int, int, int], Any]


def make_policy_factory(overrides: Dict[str, Any]) -> Tuple[PolicyFactory, Dict[str, Any]]:
    """构造策略工厂。

    默认（无 --set）：官方等价路径 entry.build_policy(BuildContext(...))，每台机器人每局各调一次。
    有 --set 覆盖：entry.CoverageRulePolicy(dict(entry.DEFAULT_CONFIG, **overrides), artifact_dir)。
    """
    if overrides:
        if not hasattr(entry, "DEFAULT_CONFIG"):
            raise SystemExit("entry 模块没有 DEFAULT_CONFIG，无法使用 --set 覆盖（impl-policy 尚未就绪？）")
        if not hasattr(entry, "CoverageRulePolicy"):
            raise SystemExit("entry 模块没有 CoverageRulePolicy，无法使用 --set 覆盖（impl-policy 尚未就绪？）")
        merged = dict(entry.DEFAULT_CONFIG)
        merged.update(overrides)

        def factory_override(agent_index: int, scenario_seed: int, repeat_index: int) -> Any:
            return entry.CoverageRulePolicy(dict(merged), ARTIFACT_DIR)

        return factory_override, {"mode": "CoverageRulePolicy", "overrides": dict(overrides)}

    def factory_default(agent_index: int, scenario_seed: int, repeat_index: int) -> Any:
        build_ctx = BuildContext(
            spec=get_protocol_spec(),
            artifact_dir=ARTIFACT_DIR,
            device="cpu",
            limits=ResourceLimits(),
            rng=np.random.default_rng(0),
        )
        return entry.build_policy(build_ctx)

    return factory_default, {"mode": "build_policy", "overrides": {}}


# ---------------------------------------------------------------------------
# 单局执行
# ---------------------------------------------------------------------------


def run_episode(
    *,
    group_id: str,
    case_id: str,
    task_config: Any,
    scenario_seed: int,
    repeat_index: int,
    factory: PolicyFactory,
) -> Dict[str, Any]:
    """跑一局，返回逐局记录（含 R/J/覆盖率曲线/碰撞率/耗时）。"""
    horizon = int(task_config.horizon)
    started = time.perf_counter()

    env = None
    policies: List[Any] = []
    status = "ok"
    error = ""
    total_return = 0.0
    steps_completed = 0
    step_coverage: List[float] = []
    step_collision: List[float] = []
    act_ms: List[float] = []
    build_ms: List[float] = []
    policy_errors: List[int] = []

    try:
        env = make_training_env(task_config)
        obs, _info = env.reset(seed=int(scenario_seed))
        agent_ids = list(env.agents)
        num_agents = len(agent_ids)

        for i, _agent_id in enumerate(agent_ids):
            t0 = time.perf_counter()
            policy = factory(i, int(scenario_seed), int(repeat_index))
            build_ms.append((time.perf_counter() - t0) * 1000.0)
            context = EpisodeContext(
                agent_index=i,
                num_agents=num_agents,
                num_targets=int(task_config.num_targets),
                horizon=horizon,
                task=_public_task_params(task_config),
                policy_seed=_policy_seed(int(scenario_seed), int(repeat_index), i),
            )
            policy.reset(context)
            policies.append(policy)

        for step in range(horizon):
            actions: Dict[str, Any] = {}
            for i, agent_id in enumerate(agent_ids):
                t0 = time.perf_counter()
                action = policies[i].act(obs[agent_id])
                act_ms.append((time.perf_counter() - t0) * 1000.0)
                problem = _check_action(action)
                if problem is not None:
                    raise ProtocolViolation(f"agent {i} 第 {step} 步动作非法: {problem}")
                actions[agent_id] = action

            obs, _rewards, _terminations, _truncations, infos = env.step(actions)

            # 奖励与指标一律取官方 infos（共享团队奖励，三台机器人数值相同）
            info0 = infos[agent_ids[0]]
            terms = info0["reward_terms"]
            metrics = info0["metrics"]
            if terms is None or metrics is None:
                raise RuntimeError(f"第 {step} 步 infos 缺少 reward_terms/metrics")
            total_return += float(terms["team_reward"])
            step_coverage.append(float(metrics.coverage_rate))
            step_collision.append(float(metrics.collision_rate))
            steps_completed += 1

    except ProtocolViolation as exc:
        status = "protocol_error"
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - 逐局失败需要如实记录而不是中断整轮
        status = "load_error" if not policies else "runtime_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for i, policy in enumerate(policies):
            try:
                policy.close()
            except Exception as exc:  # noqa: BLE001
                if status == "ok":
                    status = "runtime_error"
                    error = f"close 异常 agent {i}: {type(exc).__name__}: {exc}"
            # F11：静默降级观测。策略内部捕获异常后会退回零动作，局仍为 ok，
            # 因此必须单独统计 error_count，否则"不报错但持续退化"会被漏掉。
            _stats = getattr(policy, "stats", None)
            if isinstance(_stats, dict):
                try:
                    policy_errors.append(int(_stats.get("error_count", 0)))
                except (TypeError, ValueError):
                    policy_errors.append(-1)
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                if status == "ok":
                    status = "runtime_error"
                    error = f"env.close 异常: {type(exc).__name__}: {exc}"

    if steps_completed > 0:
        mean_j = float(total_return / steps_completed) if status != "ok" else float(total_return / horizon)
    else:
        mean_j = None

    return {
        "group_id": group_id,
        "case_id": case_id,
        "scenario_seed": int(scenario_seed),
        "repeat_index": int(repeat_index),
        "status": status,
        "error": error,
        "horizon": horizon,
        "steps_completed": steps_completed,
        "return_sum": float(total_return),
        "mean_j": mean_j,
        "mean_coverage_rate": float(np.mean(step_coverage)) if step_coverage else None,
        "mean_collision_rate": float(np.mean(step_collision)) if step_collision else None,
        "step_coverage_curve": [float(v) for v in step_coverage],
        "step_collision_curve": [float(v) for v in step_collision],
        "act_ms_mean": float(np.mean(act_ms)) if act_ms else None,
        "act_ms_p95": float(np.percentile(act_ms, 95)) if act_ms else None,
        "act_ms_max": float(max(act_ms)) if act_ms else None,
        "act_ms_samples": [round(float(v), 5) for v in act_ms],
        "act_count": len(act_ms),
        "policy_error_count": int(sum(policy_errors)) if policy_errors else 0,
        "build_ms_sum": float(sum(build_ms)),
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }


# ---------------------------------------------------------------------------
# 聚合（与官方 aggregate_episodes 同构：先算 scenario 均值，再对场景取平均）
# ---------------------------------------------------------------------------


def _scenario_summary(
    records: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """按 scenario_seed 汇总：先对 repeat 取均值。"""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(int(rec["scenario_seed"]), []).append(rec)

    summary: Dict[int, Dict[str, Any]] = {}
    for seed in sorted(grouped):
        rows = grouped[seed]
        ok_rows = [r for r in rows if r["status"] == "ok" and r["mean_j"] is not None]
        if not ok_rows:
            summary[seed] = {"n_repeats": len(rows), "n_ok": 0}
            continue
        curves = [r["step_coverage_curve"] for r in ok_rows if len(r["step_coverage_curve"]) > 0]
        col_curves = [r["step_collision_curve"] for r in ok_rows if len(r["step_collision_curve"]) > 0]
        summary[seed] = {
            "n_repeats": len(rows),
            "n_ok": len(ok_rows),
            "mean_j": float(np.mean([r["mean_j"] for r in ok_rows])),
            "mean_return": float(np.mean([r["return_sum"] for r in ok_rows])),
            "mean_coverage_rate": float(np.mean([r["mean_coverage_rate"] for r in ok_rows])),
            "mean_collision_rate": float(np.mean([r["mean_collision_rate"] for r in ok_rows])),
            "step_coverage_curve": (
                [float(v) for v in np.mean(np.asarray(curves, dtype=np.float64), axis=0)]
                if curves
                else []
            ),
            "step_collision_curve": (
                [float(v) for v in np.mean(np.asarray(col_curves, dtype=np.float64), axis=0)]
                if col_curves
                else []
            ),
        }
    return summary


def _bootstrap_indices(n_scenarios: int, group_index: int, resamples: int, seed: int) -> Optional[np.ndarray]:
    """官方同构 bootstrap 抽样索引：SeedSequence(seed, spawn_key=(group_index,))。"""
    if n_scenarios < 2:
        return None
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(group_index),)))
    return rng.integers(0, n_scenarios, size=(int(resamples), n_scenarios))


def _ci95(values: np.ndarray) -> Tuple[float, float]:
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low = float(np.percentile(values, 100.0 * alpha))
    high = float(np.percentile(values, 100.0 * (1.0 - alpha)))
    return low, high


def aggregate(
    records: Sequence[Dict[str, Any]],
    group_order: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """按组聚合：scenario 均值 -> 组均值 J_bar -> S = 500*(J_bar_basic + J_bar_coop) + CI95。"""
    per_group: Dict[str, Dict[str, Any]] = {}
    index_matrices: Dict[str, Optional[np.ndarray]] = {}

    for group_index, group_id in enumerate(group_order):
        group_records = [r for r in records if r["group_id"] == group_id]
        scenario = _scenario_summary(group_records)
        ok_seeds = [s for s in sorted(scenario) if scenario[s].get("n_ok", 0) > 0]
        scenario_means = np.asarray([scenario[s]["mean_j"] for s in ok_seeds], dtype=np.float64)

        all_act_ms = [v for r in group_records for v in [r["act_ms_p95"]] if v is not None]
        raw_act_ms = [r["act_ms_mean"] for r in group_records if r["act_ms_mean"] is not None]
        pooled_act_ms = [v for r in group_records for v in r.get("act_ms_samples", [])]
        curves = [scenario[s]["step_coverage_curve"] for s in ok_seeds if scenario[s]["step_coverage_curve"]]
        col_curves = [scenario[s]["step_collision_curve"] for s in ok_seeds if scenario[s]["step_collision_curve"]]

        entry_out: Dict[str, Any] = {
            "num_scenarios": len(scenario),
            "num_scenarios_ok": len(ok_seeds),
            "num_episodes": len(group_records),
            "num_episodes_ok": sum(1 for r in group_records if r["status"] == "ok"),
            "mean_j": float(np.mean(scenario_means)) if scenario_means.size else None,
            "mean_return": (
                float(np.mean([scenario[s]["mean_return"] for s in ok_seeds])) if ok_seeds else None
            ),
            "mean_coverage_rate": (
                float(np.mean([scenario[s]["mean_coverage_rate"] for s in ok_seeds])) if ok_seeds else None
            ),
            "mean_collision_rate": (
                float(np.mean([scenario[s]["mean_collision_rate"] for s in ok_seeds])) if ok_seeds else None
            ),
            "per_scenario_j": {str(s): scenario[s].get("mean_j") for s in sorted(scenario)},
            "step_coverage_curve": (
                [float(v) for v in np.mean(np.asarray(curves, dtype=np.float64), axis=0)] if curves else []
            ),
            "step_collision_curve": (
                [float(v) for v in np.mean(np.asarray(col_curves, dtype=np.float64), axis=0)] if col_curves else []
            ),
            "act_ms_p95_of_episode_p95": float(np.percentile(all_act_ms, 95)) if all_act_ms else None,
            "act_ms_mean_of_episode_mean": float(np.mean(raw_act_ms)) if raw_act_ms else None,
            "act_calls": len(pooled_act_ms),
            "act_ms_p95_pooled": float(np.percentile(pooled_act_ms, 95)) if pooled_act_ms else None,
            "act_ms_mean_pooled": float(np.mean(pooled_act_ms)) if pooled_act_ms else None,
            "act_ms_max_pooled": float(max(pooled_act_ms)) if pooled_act_ms else None,
            "ci95_j": None,
        }

        idx = _bootstrap_indices(len(ok_seeds), group_index, resamples, seed)
        index_matrices[group_id] = idx
        if idx is not None:
            resampled = np.mean(scenario_means[idx], axis=1)
            entry_out["ci95_j"] = [float(v) for v in _ci95(resampled)]
            entry_out["bootstrap_j_std"] = float(np.std(resampled))

        per_group[group_id] = entry_out

    # 总分与总 CI（各组 bootstrap 索引同时使用 => 联合分布）
    score: Dict[str, Any] = {"S": None, "ci95_S": None, "formula": "S = 500 * (J_bar_basic + J_bar_coop)"}
    if all(per_group.get(gid, {}).get("mean_j") is not None for gid in group_order):
        j_bars = [per_group[gid]["mean_j"] for gid in group_order]
        score["S"] = float(SCORE_SCALE * sum(j_bars))
        score["j_bar_by_group"] = {gid: per_group[gid]["mean_j"] for gid in group_order}
        if all(index_matrices.get(gid) is not None for gid in group_order):
            total = np.zeros(resamples, dtype=np.float64)
            for gid in group_order:
                group_records = [r for r in records if r["group_id"] == gid]
                scenario = _scenario_summary(group_records)
                ok_seeds = [s for s in sorted(scenario) if scenario[s].get("n_ok", 0) > 0]
                means = np.asarray([scenario[s]["mean_j"] for s in ok_seeds], dtype=np.float64)
                total += np.mean(means[index_matrices[gid]], axis=1)
            score["ci95_S"] = [float(v) for v in _ci95(SCORE_SCALE * total)]
            score["bootstrap_S_std"] = float(np.std(SCORE_SCALE * total))

    complete = all(
        per_group.get(gid, {}).get("num_episodes") == per_group.get(gid, {}).get("num_episodes_ok")
        for gid in group_order
    )
    return {"group_order": list(group_order), "groups": per_group, "score": score, "complete": bool(complete)}


# ---------------------------------------------------------------------------
# 运行计划
# ---------------------------------------------------------------------------


def build_plan(
    suite: EvaluationSuite,
    seeds_by_group: Dict[str, Sequence[int]],
    include_public: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (私有口径用例, 公开口径用例)。用例 = 一个 scenario（含 repeat 次数）。"""
    uniform_tc = suite.groups[GROUP_UNIFORM_INDEX].cases[0].task_config
    crossing_tc = suite.groups[GROUP_CROSSING_INDEX].cases[0].task_config
    uniform_gid = suite.groups[GROUP_UNIFORM_INDEX].group_id
    crossing_gid = suite.groups[GROUP_CROSSING_INDEX].group_id

    if uniform_tc.scenario.layout_kind != "uniform":
        raise SystemExit(f"groups[0] 布局不是 uniform: {uniform_tc.scenario.layout_kind}")
    if crossing_tc.scenario.layout_kind != "crossing":
        raise SystemExit(f"groups[1] 布局不是 crossing: {crossing_tc.scenario.layout_kind}")

    tc_by_layout = {"uniform": uniform_tc, "crossing": crossing_tc}
    gid_by_layout = {"uniform": uniform_gid, "crossing": crossing_gid}

    private: List[Dict[str, Any]] = []
    for layout in ("uniform", "crossing"):
        for seed in seeds_by_group[layout]:
            private.append({
                "caliber": "private",
                "group_id": gid_by_layout[layout],
                "case_id": f"{gid_by_layout[layout]}-s{int(seed)}",
                "task_config": tc_by_layout[layout],
                "scenario_seed": int(seed),
                "repeats": 1,
            })

    public: List[Dict[str, Any]] = []
    if include_public:
        for group in suite.groups:
            for case in group.cases:
                public.append({
                    "caliber": "public",
                    "group_id": group.group_id,
                    "case_id": case.case_id,
                    "task_config": case.task_config,
                    "scenario_seed": int(case.scenario_seed),
                    "repeats": int(group.policy_repeats),
                })
    return private, public


def run_plan(cases: Sequence[Dict[str, Any]], factory: PolicyFactory, block_label: str = "A") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case in cases:
        for repeat_index in range(int(case["repeats"])):
            record = run_episode(
                group_id=case["group_id"],
                case_id=case["case_id"],
                task_config=case["task_config"],
                scenario_seed=case["scenario_seed"],
                repeat_index=repeat_index,
                factory=factory,
            )
            record["caliber"] = case["caliber"]
            record["seed_block"] = block_label
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_ci(ci: Optional[Sequence[float]], digits: int = 3) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def print_caliber(caliber: str, result: Dict[str, Any]) -> None:
    print(f"\n--- 口径 {caliber} ---")
    header = (
        f"{'group':<14}{'scen':>5}{'ep':>5}{'J_bar':>10}{'cov':>9}{'col':>9}"
        f"{'act_p95(ms)':>13}{'CI95(J_bar)':>26}"
    )
    print(header)
    for gid in result["group_order"]:
        g = result["groups"][gid]
        print(
            f"{gid:<14}{g['num_scenarios']:>5}{g['num_episodes']:>5}"
            f"{_fmt(g['mean_j']):>10}{_fmt(g['mean_coverage_rate']):>9}{_fmt(g['mean_collision_rate']):>9}"
            f"{_fmt(g['act_ms_p95_pooled'], 3):>13}{_fmt_ci(g['ci95_j']):>26}"
        )
    score = result["score"]
    print(
        f"SCORE {caliber}: S = {_fmt(score['S'], 3)}   CI95(S) = {_fmt_ci(score['ci95_S'], 3)}"
        f"   complete={result['complete']}"
    )
    for gid in result["group_order"]:
        curve = result["groups"][gid]["step_coverage_curve"]
        if curve:
            print(f"每步平均覆盖率曲线 {caliber}/{gid}: " + " ".join(f"{v:.3f}" for v in curve))


def print_episode_failures(records: Sequence[Dict[str, Any]]) -> None:
    bad = [r for r in records if r["status"] != "ok"]
    if not bad:
        return
    print(f"\n!!! 非正常局 {len(bad)}/{len(records)}：")
    for r in bad:
        print(
            f"  caliber={r.get('caliber')} group={r['group_id']} case={r['case_id']} "
            f"seed={r['scenario_seed']} repeat={r['repeat_index']} status={r['status']} "
            f"steps={r['steps_completed']} error={r['error']}"
        )


CSV_FIELDS = [
    "caliber", "seed_block", "group_id", "case_id", "scenario_seed", "repeat_index", "status", "error",
    "horizon", "steps_completed", "return_sum", "mean_j", "mean_coverage_rate",
    "mean_collision_rate", "act_ms_mean", "act_ms_p95", "act_ms_max", "act_count",
    "build_ms_sum", "wall_ms",
]


def write_outputs(
    output_dir: Path,
    tag: str,
    payload: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"mc-{tag}.json"
    csv_path = output_dir / f"mc-{tag}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_override(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set 需要 key=value 形式: {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--set 的键为空: {text!r}")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        value: Any = lowered == "true"
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    return key, value


def _parse_seed_list(text: str) -> List[int]:
    seeds = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not seeds:
        raise argparse.ArgumentTypeError("种子列表为空")
    return seeds


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P666 多种子蒙特卡洛评测（官方环境，双口径）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True, help="输出目录（如 outputs\\P666\\E001）")
    parser.add_argument("--n-per-group", type=int, default=50, help="私有口径每组场景数（取固定种子表前 n 个）")
    parser.add_argument("--seed-block", type=str, default="A", choices=["A", "B", "C", "D", "both"],
                        help="私有口径的固定种子块：A（默认，历史口径）、B（跨块复现）、C/D（F2 场景池）、both（同时给出 A 与 B）")
    parser.add_argument("--inject-error-probe", type=int, default=0,
                        help="自检开关（默认 0=关闭）：让前 n 个策略实例先经历一次内部异常，用于证明 F11 断言会触发")
    parser.add_argument("--seeds-uniform", type=str, default=None, help="逗号分隔的 uniform 种子，覆盖 n-per-group 与 --seed-block")
    parser.add_argument("--seeds-crossing", type=str, default=None, help="逗号分隔的 crossing 种子，覆盖 n-per-group 与 --seed-block")
    parser.add_argument("--set", dest="overrides", action="append", default=[], type=_parse_override,
                        help="覆盖 entry.DEFAULT_CONFIG 的 key=value，可重复（走 CoverageRulePolicy）")
    parser.add_argument("--tag", type=str, default="default", help="输出文件名标签 mc-<tag>.json/csv")
    parser.add_argument("--include-public", action="store_true", help="包含公开 4 例口径（默认包含）")
    parser.add_argument("--no-public", action="store_true", help="只跑私有种子口径")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES, help="bootstrap 重采样次数")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED, help="bootstrap 随机种子")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001 - 指纹缺失不应中断评测
        return None


def _policy_provenance() -> Dict[str, Any]:
    """记录本次运行实际使用的 entry.py 指纹：换版本后数字不可直接横向比较。"""
    raw = getattr(entry, "__file__", None)
    info: Dict[str, Any] = {"entry_path": None, "entry_sha256": None, "entry_size_bytes": None, "entry_mtime": None}
    if not raw:
        return info
    path = Path(raw).resolve()
    info["entry_path"] = str(path)
    if path.is_file():
        info["entry_sha256"] = _file_sha256(path)
        stat = path.stat()
        info["entry_size_bytes"] = int(stat.st_size)
        info["entry_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
    return info


def _cross_block_summary(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> Dict[str, Any]:
    """跨块差异：S、CI95 是否重叠、逐组 J_bar 差（A - B）。"""
    score_a, score_b = res_a["score"], res_b["score"]
    out: Dict[str, Any] = {
        "block_a": res_a.get("block"),
        "block_b": res_b.get("block"),
        "S_a": score_a.get("S"),
        "S_b": score_b.get("S"),
        "delta_S": (
            score_a["S"] - score_b["S"]
            if score_a.get("S") is not None and score_b.get("S") is not None else None
        ),
        "ci95_a": score_a.get("ci95_S"),
        "ci95_b": score_b.get("ci95_S"),
        "ci_overlap": None,
        "per_group_delta": {},
    }
    if out["ci95_a"] and out["ci95_b"]:
        out["ci_overlap"] = bool(
            out["ci95_a"][0] <= out["ci95_b"][1] and out["ci95_b"][0] <= out["ci95_a"][1]
        )
    for gid in res_a["group_order"]:
        j_a = res_a["groups"].get(gid, {}).get("mean_j")
        j_b = res_b["groups"].get(gid, {}).get("mean_j")
        out["per_group_delta"][gid] = (j_a - j_b) if (j_a is not None and j_b is not None) else None
    return out


def print_cross_block(summary: Dict[str, Any]) -> None:
    delta = summary["delta_S"]
    per_group = "  ".join(
        f"{gid}:{('%+.4f' % d) if d is not None else 'n/a'}"
        for gid, d in summary["per_group_delta"].items()
    )
    print(
        "跨块差异: "
        f"S_A={_fmt(summary['S_a'], 3)}  S_B={_fmt(summary['S_b'], 3)}  "
        f"ΔS(A-B)={('%+.3f' % delta) if delta is not None else 'n/a'}  "
        f"CI95_A={_fmt_ci(summary['ci95_a'])}  CI95_B={_fmt_ci(summary['ci95_b'])}  "
        f"CI 重叠={'是' if summary['ci_overlap'] else '否'}  "
        f"各组J_bar差(A-B) {per_group}"
    )


def _wrap_factory_with_error_probe(factory: PolicyFactory, n: int) -> PolicyFactory:
    """F11 自检开关：让前 n 个策略实例先经历一次内部异常（entry.act 会吞掉并只计数）。

    仅用于**证明**静默降级断言真的会触发；默认关闭（n=0）。不修改 entry.py，也不改变正常路径：
    注入发生在策略 reset 之后（reset 会清空计数器）、第一次真实 act 之前，异常由 entry.act 内部捕获。
    """
    left = {"n": int(n)}

    def wrapped(agent_index: int, scenario_seed: int, repeat_index: int):
        policy = factory(agent_index, scenario_seed, repeat_index)
        if left["n"] > 0:
            left["n"] -= 1
            original_reset = policy.reset

            def reset_then_inject(context, _orig=original_reset, _pol=policy):
                _orig(context)
                try:
                    _pol.act({})  # 空观测 => entry._act_impl 抛异常 => error_count += 1
                except Exception:  # noqa: BLE001 - 注入失败不影响正常评测
                    pass

            policy.reset = reset_then_inject
        return policy

    return wrapped


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_stdout()
    args = parse_args(argv)
    if args.n_per_group <= 0:
        print("!! --n-per-group 必须为正整数")
        return 2
    overrides = dict(args.overrides)
    include_public = not args.no_public

    suite = load_suite(SUITE_PATH)
    group_order = [g.group_id for g in suite.groups]

    custom = args.seeds_uniform is not None or args.seeds_crossing is not None
    # (caliber_key, block_label, {布局组: 种子列表}, 说明文本)
    private_specs: List[Tuple[str, str, Dict[str, List[int]], str]] = []
    if custom:
        fallback_block = args.seed_block if args.seed_block in ("A", "B") else "A"
        seeds = {
            "uniform": _parse_seed_list(args.seeds_uniform) if args.seeds_uniform is not None
            else list(seedset.list_seeds(fallback_block, seedset.GROUP_UNIFORM, args.n_per_group)),
            "crossing": _parse_seed_list(args.seeds_crossing) if args.seeds_crossing is not None
            else list(seedset.list_seeds(fallback_block, seedset.GROUP_CROSSING, args.n_per_group)),
        }
        private_specs.append(("private", "custom", seeds, f"自定义种子（缺省侧回退块{fallback_block}）"))
    else:
        block_list = ["A", "B"] if args.seed_block == "both" else [args.seed_block]
        for position, block in enumerate(block_list):
            seeds = {
                "uniform": list(seedset.list_seeds(block, seedset.GROUP_UNIFORM, args.n_per_group)),
                "crossing": list(seedset.list_seeds(block, seedset.GROUP_CROSSING, args.n_per_group)),
            }
            key = "private" if position == 0 else f"private-{block}"
            private_specs.append((key, block, seeds, seedset.describe_block(block)))

    factory, policy_info = make_policy_factory(overrides)
    if args.inject_error_probe > 0:
        factory = _wrap_factory_with_error_probe(factory, args.inject_error_probe)
        print(f"!! 自检模式：已对前 {args.inject_error_probe} 个策略实例注入内部异常（预期断言会失败）")
    provenance = _policy_provenance()

    print("=== P666 MC 评测 ===")
    print(f"仓库根        : {REPO_ROOT}")
    print(f"提交目录      : {SUBMISSION_DIR}")
    print(f"官方套件      : {SUITE_PATH.name}  suite_id={suite.suite_id}  hash={suite.suite_hash[:16]}")
    print(f"策略模式      : {policy_info['mode']}  overrides={policy_info['overrides']}")
    print(f"entry.py      : sha256={provenance['entry_sha256']}  mtime={provenance['entry_mtime']}")
    print(f"种子块选择    : --seed-block {args.seed_block}{'（自定义种子优先）' if custom else ''}")
    for key, block, seeds, text in private_specs:
        print(
            f"私有口径[{key}] : {text}；uniform {len(seeds['uniform'])} 例 / crossing "
            f"{len(seeds['crossing'])} 例（每例 1 局）"
        )
    print(f"公开口径      : {'包含（4 例 × policy_repeats=2 = 8 局）' if include_public else '已关闭'}")

    started = time.perf_counter()
    all_records: List[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    caliber_labels: Dict[str, str] = {}

    for key, block, seeds, _text in private_specs:
        private_cases, _ = build_plan(suite, seeds, False)
        records = run_plan(private_cases, factory, block_label=block)
        result = aggregate(records, group_order, resamples=args.bootstrap, seed=args.bootstrap_seed)
        result["block"] = block
        result["seeds"] = {"uniform": list(seeds["uniform"]), "crossing": list(seeds["crossing"])}
        results[key] = result
        caliber_labels[key] = f"{key}·块{block}" if block in ("A", "B") else f"{key}·{block}"
        all_records.extend(records)

    if include_public:
        _, public_cases = build_plan(suite, private_specs[0][2], True)
        public_records = run_plan(public_cases, factory, block_label="public")
        public_result = aggregate(public_records, group_order, resamples=args.bootstrap, seed=args.bootstrap_seed)
        public_result["block"] = "public"
        results["public"] = public_result
        all_records.extend(public_records)

    wall_s = time.perf_counter() - started

    for key, _block, _seeds, _text in private_specs:
        print_caliber(caliber_labels[key], results[key])
    if "public" in results:
        print_caliber("public", results["public"])

    cross_block = None
    if "private" in results and "private-B" in results:
        cross_block = _cross_block_summary(results["private"], results["private-B"])
        print_cross_block(cross_block)

    print_episode_failures(all_records)

    # F11：静默降级断言。策略内部吞掉异常后会退回零动作，局仍为 ok，
    # 只看 status 会漏掉"不报错但持续退化"，因此这里单独统计 error_count。
    n_policy_errors = sum(int(r.get("policy_error_count", 0) or 0) for r in all_records)
    if n_policy_errors:
        print(f"!! 策略静默降级：policy_error_count 合计 = {n_policy_errors}")
        for r in all_records:
            if int(r.get("policy_error_count", 0) or 0):
                print(f"   {r['case_id']} repeat={r['repeat_index']} seed={r['scenario_seed']} "
                      f"error_count={r['policy_error_count']}")
    else:
        print("策略异常计数断言：全部为 0（无静默降级）")

    n_failed = sum(1 for r in all_records if r["status"] != "ok")
    print(f"\n总用时 {wall_s:.1f}s；局数 {len(all_records)}；非正常局 {n_failed}；"
          f"策略内部异常 {n_policy_errors}")

    payload: Dict[str, Any] = {
        "tool": "eval_mc",
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repo_root": str(REPO_ROOT),
        "submission_dir": str(SUBMISSION_DIR),
        "suite": {"path": str(SUITE_PATH), "suite_id": suite.suite_id, "suite_hash": suite.suite_hash},
        "policy": policy_info,
        "entry": provenance,
        "seed_block": "custom" if custom else args.seed_block,
        "private_calibers": {
            key: {"block": block, "describe": text}
            for key, block, _seeds, text in private_specs
        },
        # 兼容旧字段：primary 私有口径的种子列表
        "seeds": {
            "uniform": list(private_specs[0][2]["uniform"]),
            "crossing": list(private_specs[0][2]["crossing"]),
        },
        "bootstrap": {"resamples": int(args.bootstrap), "seed": int(args.bootstrap_seed), "confidence": CONFIDENCE_LEVEL},
        "wall_s": wall_s,
        "n_episodes": len(all_records),
        "n_failed": n_failed,
        "n_policy_internal_errors": n_policy_errors,
        "calibers": results,
        "cross_block": cross_block,
        "episodes": all_records,
    }
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    json_path, csv_path = write_outputs(out_dir, args.tag, payload, all_records)
    print(f"已写出: {json_path}")
    print(f"已写出: {csv_path}")

    return 0 if (n_failed == 0 and n_policy_errors == 0) else 3


if __name__ == "__main__":
    raise SystemExit(main())
