"""P666 策略鲁棒性自测：极端输入 / 跨局状态 / 动作契约。

每条检查打印 PASS 或 FAIL。动作契约（dtype=float32、shape=(2,)、有限、∈[-1,1]）在每条 act 上
都做断言 —— 官方不裁剪非法动作，一旦越界该局就是 protocol_error，整次评分为 null。

检查项:
  1. 全不可见观测（实体存在但不可见；以及实体完全不存在）
  2. 目标与机器人精确重合（相对位置 rel=0，可见标记仍为真）
  3. 观测含 NaN / inf
  4. step_index 复位与跳变
  5. 同一策略实例连续 8 局复用（reset 是否清空跨局状态）
  6. close() 不抛异常

用法:
    .venv\\Scripts\\python.exe participant\\P666\\tools_local\\robustness_check.py
    .venv\\Scripts\\python.exe participant\\P666\\tools_local\\robustness_check.py --set safety_radius=0.0
"""

from __future__ import annotations

import argparse
import sys
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

from coverage_bench.protocol import (  # noqa: E402
    BuildContext,
    EpisodeContext,
    PublicTaskParams,
    ResourceLimits,
    get_protocol_spec,
)
from coverage_bench.suites import load_suite  # noqa: E402

import entry  # noqa: E402  # 参赛策略（静态导入）

A = 8  # agent_capacity
B = 8  # target_capacity
HORIZON = 10
FIXED_POLICY_SEED = 424242


def _force_utf8_stdout() -> None:
    """Windows 控制台默认代码页会让中文输出变成乱码；这里显式切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 输出流可能已被重定向且不可重配置
            pass


# ---------------------------------------------------------------------------
# 观测构造
# ---------------------------------------------------------------------------


def base_obs(
    agent_index: int = 0,
    step_index: int = 0,
    horizon: int = HORIZON,
    *,
    self_state: Optional[Sequence[float]] = None,
    peers: Optional[np.ndarray] = None,
    peer_exists: Optional[np.ndarray] = None,
    peer_visible: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    target_exists: Optional[np.ndarray] = None,
    target_visible: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """构造一个结构合法的 AgentObservation（键/dtype/shape 与官方一致）。"""
    self_arr = np.array(self_state if self_state is not None else (0.0, 0.0, 0.0, 0.0, 0.05), dtype=np.float32)
    peers_arr = np.zeros((A, 5), dtype=np.float32) if peers is None else np.asarray(peers, dtype=np.float32)
    peer_exists_arr = np.zeros(A, dtype=np.bool_) if peer_exists is None else np.asarray(peer_exists, dtype=np.bool_)
    peer_visible_arr = np.zeros(A, dtype=np.bool_) if peer_visible is None else np.asarray(peer_visible, dtype=np.bool_)
    targets_arr = np.zeros((B, 3), dtype=np.float32) if targets is None else np.asarray(targets, dtype=np.float32)
    target_exists_arr = np.zeros(B, dtype=np.bool_) if target_exists is None else np.asarray(target_exists, dtype=np.bool_)
    target_visible_arr = np.zeros(B, dtype=np.bool_) if target_visible is None else np.asarray(target_visible, dtype=np.bool_)

    return {
        "self_state": np.ascontiguousarray(self_arr),
        "peers": np.ascontiguousarray(peers_arr),
        "peer_exists": np.ascontiguousarray(peer_exists_arr),
        "peer_visible": np.ascontiguousarray(peer_visible_arr),
        "targets": np.ascontiguousarray(targets_arr),
        "target_exists": np.ascontiguousarray(target_exists_arr),
        "target_visible": np.ascontiguousarray(target_visible_arr),
        "agent_index": np.int64(agent_index),
        "step_index": np.int64(step_index),
        "time_remaining": np.float32((horizon - step_index) / horizon),
    }


def exists_flags(num_agents: int = 3, num_targets: int = 3, agent_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    peer_exists = np.zeros(A, dtype=np.bool_)
    peer_exists[:num_agents] = True
    peer_exists[agent_index] = False
    target_exists = np.zeros(B, dtype=np.bool_)
    target_exists[:num_targets] = True
    return peer_exists, target_exists


# ---------------------------------------------------------------------------
# 报告与动作契约
# ---------------------------------------------------------------------------


class Reporter:
    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        self.results.append((name, ok, detail))

    @property
    def failed(self) -> List[Tuple[str, bool, str]]:
        return [r for r in self.results if not r[1]]


def action_contract(action: Any) -> Optional[str]:
    """返回 None 表示合法；否则返回违规描述。"""
    if not isinstance(action, np.ndarray):
        return f"不是 ndarray: {type(action).__name__}"
    if action.dtype != np.float32:
        return f"dtype={action.dtype}"
    if action.shape != (2,):
        return f"shape={action.shape}"
    if not np.isfinite(action).all():
        return f"含非有限值: {action.tolist()}"
    if np.any(action < -1.0) or np.any(action > 1.0):
        return f"越界: {action.tolist()}"
    return None


def try_act(policy: Any, obs: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """调用 act 并做契约检查；返回 (action, 问题描述)。"""
    try:
        action = policy.act(obs)
    except Exception as exc:  # noqa: BLE001 - 这里就是要记录策略的具体失败
        return None, f"act 抛异常 {type(exc).__name__}: {exc}"
    problem = action_contract(action)
    if problem is not None:
        return None, f"动作违规: {problem}"
    return action, None


# ---------------------------------------------------------------------------
# 策略工厂（与 eval_mc.py 保持同一口径）
# ---------------------------------------------------------------------------


def make_policy_factory(overrides: Dict[str, Any]) -> Tuple[Callable[[int], Any], Dict[str, Any]]:
    if overrides:
        if not hasattr(entry, "DEFAULT_CONFIG"):
            raise SystemExit("entry 模块没有 DEFAULT_CONFIG，无法使用 --set 覆盖")
        if not hasattr(entry, "CoverageRulePolicy"):
            raise SystemExit("entry 模块没有 CoverageRulePolicy，无法使用 --set 覆盖")
        merged = dict(entry.DEFAULT_CONFIG)
        merged.update(overrides)

        def factory_override(agent_index: int) -> Any:
            return entry.CoverageRulePolicy(dict(merged), ARTIFACT_DIR)

        return factory_override, {"mode": "CoverageRulePolicy", "overrides": dict(overrides)}

    def factory_default(agent_index: int) -> Any:
        build_ctx = BuildContext(
            spec=get_protocol_spec(),
            artifact_dir=ARTIFACT_DIR,
            device="cpu",
            limits=ResourceLimits(),
            rng=np.random.default_rng(0),
        )
        return entry.build_policy(build_ctx)

    return factory_default, {"mode": "build_policy", "overrides": {}}


def _public_task_params(task_config: Any) -> PublicTaskParams:
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


def episode_context(task: PublicTaskParams, agent_index: int = 0, policy_seed: int = FIXED_POLICY_SEED) -> EpisodeContext:
    return EpisodeContext(
        agent_index=agent_index,
        num_agents=3,
        num_targets=3,
        horizon=HORIZON,
        task=task,
        policy_seed=int(policy_seed),
    )


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------


def check_all_invisible(reporter: Reporter, factory: Callable[[int], Any], task: PublicTaskParams) -> None:
    policy = factory(0)
    policy.reset(episode_context(task))

    peer_exists, target_exists = exists_flags()
    obs = base_obs(peer_exists=peer_exists, target_exists=target_exists)
    _action, problem = try_act(policy, obs)
    reporter.check("全不可见观测（3 队友 / 3 目标存在但不可见）", problem is None, problem or "动作合法")

    peer_none = np.zeros(A, dtype=np.bool_)
    target_none = np.zeros(B, dtype=np.bool_)
    obs_empty = base_obs(peer_exists=peer_none, target_exists=target_none)
    _action, problem2 = try_act(policy, obs_empty)
    reporter.check("空场景（无队友 / 无目标存在）", problem2 is None, problem2 or "动作合法")
    policy.close()


def check_exact_overlap(reporter: Reporter, factory: Callable[[int], Any], task: PublicTaskParams) -> None:
    policy = factory(0)
    policy.reset(episode_context(task))
    peer_exists, target_exists = exists_flags()

    targets = np.zeros((B, 3), dtype=np.float32)
    targets[0] = (0.0, 0.0, 0.15)  # 目标中心与机器人中心精确重合
    target_visible = np.zeros(B, dtype=np.bool_)
    target_visible[0] = True
    obs = base_obs(peer_exists=peer_exists, target_exists=target_exists,
                   targets=targets, target_visible=target_visible)
    _action, problem = try_act(policy, obs)
    reporter.check("目标与机器人精确重合 rel=(0,0) 且可见", problem is None, problem or "动作合法")

    peers = np.zeros((A, 5), dtype=np.float32)
    peers[1] = (0.0, 0.0, 0.0, 0.0, 0.05)  # 队友与机器人中心精确重合
    peer_visible = np.zeros(A, dtype=np.bool_)
    peer_visible[1] = True
    obs2 = base_obs(peer_exists=peer_exists, target_exists=target_exists,
                    peers=peers, peer_visible=peer_visible)
    _action2, problem2 = try_act(policy, obs2)
    reporter.check("队友与机器人精确重合 rel=(0,0) 且可见", problem2 is None, problem2 or "动作合法")

    obs3 = base_obs(peer_exists=peer_exists, target_exists=target_exists,
                    peers=peers, peer_visible=peer_visible,
                    targets=targets, target_visible=target_visible)
    _action3, problem3 = try_act(policy, obs3)
    reporter.check("队友与目标同时精确重合", problem3 is None, problem3 or "动作合法")
    policy.close()


def check_nan_inf(reporter: Reporter, factory: Callable[[int], Any], task: PublicTaskParams) -> None:
    policy = factory(0)
    policy.reset(episode_context(task))
    peer_exists, target_exists = exists_flags()

    nan = float("nan")
    inf = float("inf")

    targets = np.zeros((B, 3), dtype=np.float32)
    targets[0] = (nan, inf, 0.15)
    targets[1] = (-inf, nan, 0.15)
    target_visible = np.zeros(B, dtype=np.bool_)
    target_visible[0] = True
    target_visible[1] = True
    obs = base_obs(self_state=(nan, 0.0, 0.0, 0.0, 0.05),
                   peer_exists=peer_exists, target_exists=target_exists,
                   targets=targets, target_visible=target_visible)
    _action, problem = try_act(policy, obs)
    reporter.check("观测含 NaN/inf（自身位置 + 目标行）", problem is None, problem or "动作合法")

    peers = np.zeros((A, 5), dtype=np.float32)
    peers[1] = (nan, 0.0, inf, 0.0, 0.05)
    peer_visible = np.zeros(A, dtype=np.bool_)
    peer_visible[1] = True
    obs2 = base_obs(peer_exists=peer_exists, target_exists=target_exists,
                    peers=peers, peer_visible=peer_visible)
    _action2, problem2 = try_act(policy, obs2)
    reporter.check("观测含 NaN/inf（队友相对量）", problem2 is None, problem2 or "动作合法")
    policy.close()


def check_step_index(reporter: Reporter, factory: Callable[[int], Any], task: PublicTaskParams) -> None:
    policy = factory(0)
    policy.reset(episode_context(task))
    peer_exists, target_exists = exists_flags()
    make = lambda step: base_obs(step_index=step, peer_exists=peer_exists, target_exists=target_exists)  # noqa: E731

    problems: List[str] = []
    for step in (0, 9, 0, 5, 0):
        _action, problem = try_act(policy, make(step))
        if problem is not None:
            problems.append(f"step={step}: {problem}")
    reporter.check("step_index 跳变（0 -> 9 -> 0 -> 5 -> 0）", not problems, "; ".join(problems) or "全部动作合法")

    policy.reset(episode_context(task))
    _action, problem = try_act(policy, make(0))
    reporter.check("reset 后 step_index 复位到 0", problem is None, problem or "动作合法")
    policy.close()


def check_instance_reuse(reporter: Reporter, factory: Callable[[int], Any], task: PublicTaskParams) -> None:
    policy = factory(0)
    peer_exists, target_exists = exists_flags()

    def obs_sequence() -> List[Dict[str, Any]]:
        out = []
        for step in range(6):
            targets = np.zeros((B, 3), dtype=np.float32)
            targets[0] = (0.3 - 0.05 * step, -0.2, 0.15)
            target_visible = np.zeros(B, dtype=np.bool_)
            target_visible[0] = True
            out.append(base_obs(step_index=step, peer_exists=peer_exists, target_exists=target_exists,
                                targets=targets, target_visible=target_visible))
        return out

    episodes: List[List[np.ndarray]] = []
    problems: List[str] = []
    for _ep in range(8):
        policy.reset(episode_context(task))
        actions: List[np.ndarray] = []
        for obs in obs_sequence():
            action, problem = try_act(policy, obs)
            if problem is not None:
                problems.append(problem)
                actions.append(np.zeros(2, dtype=np.float32))
            else:
                actions.append(action)  # type: ignore[arg-type]
        episodes.append(actions)

    if problems:
        reporter.check("同一实例连续 8 局复用（动作契约）", False, problems[0])
    else:
        reporter.check("同一实例连续 8 局复用（动作契约）", True, "8 局 × 6 步动作全部合法")

    max_diff = 0.0
    for ep in range(1, 8):
        for a, b in zip(episodes[0], episodes[ep]):
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    reporter.check(
        "reset 清空跨局状态（第 1 局与第 2..8 局同观测动作一致）",
        max_diff == 0.0,
        f"max|Δa| = {max_diff:.3e}",
    )
    policy.close()


def check_close(reporter: Reporter, factory: Callable[[int], Any]) -> None:
    try:
        policy = factory(0)
        policy.close()
        reporter.check("close() 不抛异常", True, "ok")
    except Exception as exc:  # noqa: BLE001
        reporter.check("close() 不抛异常", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_override(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set 需要 key=value 形式: {text!r}")
    key, raw = text.split("=", 1)
    key, raw = key.strip(), raw.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--set 的键为空: {text!r}")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    for caster in (int, float):
        try:
            return key, caster(raw)
        except ValueError:
            continue
    return key, raw


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P666 策略鲁棒性自测", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--set", dest="overrides", action="append", default=[], type=_parse_override,
                        help="覆盖 entry.DEFAULT_CONFIG 的 key=value，可重复")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_stdout()
    args = parse_args(argv)
    overrides = dict(args.overrides)
    factory, policy_info = make_policy_factory(overrides)

    suite = load_suite(SUITE_PATH)
    task = _public_task_params(suite.groups[0].cases[0].task_config)

    print("=== P666 鲁棒性自测 ===")
    print(f"策略模式: {policy_info['mode']}  overrides={policy_info['overrides']}")
    print(f"entry 文件: {Path(entry.__file__).resolve()}")
    print()

    reporter = Reporter()
    check_all_invisible(reporter, factory, task)
    check_exact_overlap(reporter, factory, task)
    check_nan_inf(reporter, factory, task)
    check_step_index(reporter, factory, task)
    check_instance_reuse(reporter, factory, task)
    check_close(reporter, factory)

    total = len(reporter.results)
    failed = reporter.failed
    print(f"\n合计 {total} 条：PASS {total - len(failed)} / FAIL {len(failed)}")
    for name, _ok, detail in failed:
        print(f"  FAIL {name}  [{detail}]")
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
