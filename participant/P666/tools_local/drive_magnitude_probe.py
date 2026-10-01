r"""驱动力幅度探针：动作箱 [-1,1]^2 的"对角线红利"到底有多大？

协议只说两个分量各自在 [-1,1]，驱动力 = drive_force * action（physics.py:52/79）。
因此 |u| 的最大值是 sqrt(2) 而不是 1：对角驱动比"单位向量驱动"快 41%。
由于 10 步内总位移只有约 0.25 m，而覆盖判定半径 0.15 m，可达区域面积 ∝ (位移+0.15)^2，
这 41% 会放大成约 1.5 倍的可达目标对数——这很可能是官方规则基线（175）远高于
"单位向量追踪"（约 114）的原因。

本脚本用官方环境实测：
  1) 不同动作幅度下的 10 步累计位移；
  2) 三种控制律在 oracle 分配下的 J 与 S。

用法（仓库根目录，评测环境）：
    .venv\\Scripts\\python.exe participant\\P666\\tools_local\\drive_magnitude_probe.py
"""
from __future__ import annotations

import itertools
import sys

sys.dont_write_bytecode = True  # PYC_GUARD：不写出 __pycache__（官方预检把被忽略文件也算未提交内容）
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # 仓库根: tools_local -> P666 -> participant -> <repo>
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

HORIZON = 10
N_AGENTS = 3
N_TARGETS = 3


def measure_displacement(action: np.ndarray, steps: int = 10) -> float:
    suite = load_suite(REPO / "configs" / "public-suite-v1.yaml")
    case = suite.groups[0].cases[0]
    env = make_training_env(case.task_config)
    env.reset(seed=1001)
    agents = list(env.agents)
    start = np.array(env._current_snapshot.robot_positions[0], dtype=float).copy()
    for _ in range(steps):
        if not env.agents:
            break
        acts = {a: (np.asarray(action, dtype=np.float32) if i == 0 else np.zeros(2, dtype=np.float32))
                for i, a in enumerate(agents)}
        env.step(acts)
    end = np.array(env._current_snapshot.robot_positions[0], dtype=float)
    env.close()
    return float(np.linalg.norm(end - start))


def control(rel: np.ndarray, mode: str, K: float = 20.0) -> np.ndarray:
    n = float(np.linalg.norm(rel))
    if n < 1e-9:
        return np.zeros(2)
    if mode == "unit":              # 单位向量：|u| = 1
        return rel / n
    if mode == "corner":            # 分量饱和：|u| 最大到 sqrt(2)，方向量化到 45 度
        return np.clip(K * rel, -1.0, 1.0)
    if mode == "sqrt2":             # 保方向、取最大幅度（先放大到 sqrt(2) 再按轴裁剪）
        return np.clip(np.sqrt(2.0) * rel / n, -1.0, 1.0)
    raise ValueError(mode)


def run_episode(case, seed: int, assign, mode: str, safety: float = 0.16,
                lead_gain: float = 0.0, oracle_lead: bool = False) -> tuple[float, float, float]:
    env = make_training_env(case.task_config)
    env.reset(seed=seed)
    agents = list(env.agents)
    R, covs, cols = 0.0, [], []
    for _ in range(HORIZON):
        snap = env._current_snapshot
        acts = {}
        for i, a in enumerate(agents):
            j = assign[i]
            if j is None:
                u = np.zeros(2)
            else:
                rel = np.array(snap.target_positions[j], dtype=float) - np.array(snap.robot_positions[i], dtype=float)
                if oracle_lead:
                    rel = rel + np.array(snap.target_velocities[j], dtype=float) * lead_gain
                u = control(rel, mode)
                for k in range(N_AGENTS):
                    if k == i:
                        continue
                    d = float(np.linalg.norm(np.array(snap.robot_positions[k]) - np.array(snap.robot_positions[i])))
                    if d < safety:
                        away = np.array(snap.robot_positions[i]) - np.array(snap.robot_positions[k])
                        u = u - (safety - d) / safety * 4.0 * (away / max(d, 1e-6))
            acts[a] = np.clip(np.asarray(u, dtype=float), -1.0, 1.0).astype(np.float32)
        _o, rew, _t, _tr, infos = env.step(acts)
        a0 = agents[0]
        R += rew[a0]
        covs.append(infos[a0]["reward_terms"]["coverage"])
        cols.append(-infos[a0]["reward_terms"]["collision"])
    env.close()
    return R, float(np.mean(covs)), float(np.mean(cols))


def best_of(case, seed: int, mode: str, oracle_lead: bool = False, lead_gain: float = 0.25):
    options = list(itertools.product(list(range(N_TARGETS)) + [None], repeat=N_AGENTS))
    best = (-1e9, None)
    for assign in options:
        R, _c, _k = run_episode(case, seed, assign, mode, lead_gain=lead_gain, oracle_lead=oracle_lead)
        if R > best[0]:
            best = (R, assign)
    return best[0]


def main() -> None:
    print("=" * 84)
    print("1) 动作幅度 -> 10 步累计位移（官方环境实测，seed=1001，机器人 0）")
    print("=" * 84)
    cases = [
        ("[1.0, 0.0]  |u|=1.000", np.array([1.0, 0.0])),
        ("[0.5, 0.0]  |u|=0.500", np.array([0.5, 0.0])),
        ("[1.0, 1.0]  |u|=1.414", np.array([1.0, 1.0])),
        ("[1.0, 0.5]  |u|=1.118", np.array([1.0, 0.5])),
        ("[0.7, 0.7]  |u|=0.990", np.array([0.7, 0.7])),
    ]
    for label, act in cases:
        d = measure_displacement(act)
        print(f"  {label} -> 累计位移 {d:.4f} m")

    print("\n" + "=" * 84)
    print("2) 控制律对比（oracle 最优固定分配，真实环境）")
    print("=" * 84)
    suite = load_suite(REPO / "configs" / "public-suite-v1.yaml")
    groups = [("basic(uniform)", suite.groups[0].cases[0], [1001, 1002] + list(range(5001, 5031))),
              ("coop(crossing)", suite.groups[1].cases[0], [2001, 2002] + list(range(6001, 6031)))]
    for mode in ("unit", "corner", "sqrt2"):
        for oracle_lead in (False, True):
            js = {}
            for name, case, seeds in groups:
                vals = [best_of(case, s, mode, oracle_lead=oracle_lead) / HORIZON for s in seeds]
                js[name] = float(np.mean(vals))
            S = 500.0 * (js["basic(uniform)"] + js["coop(crossing)"])
            tag = "+提前量" if oracle_lead else ""
            print(f"  mode={mode:6s}{tag:6s} J_basic={js['basic(uniform)']:.4f} "
                  f"J_coop={js['coop(crossing)']:.4f}  ==> S ≈ {S:6.1f}")
    print("\n  对照：官方 P901 random 58.33 | P902 rule 175.00 | P903 PPO(1M) 116.67")

    print("\n" + "=" * 84)
    print("3) 就近分配（可实现的规则策略）+ 三种控制律")
    print("=" * 84)
    for mode in ("unit", "corner", "sqrt2"):
        js = {}
        for name, case, seeds in groups:
            vals = []
            for s in seeds:
                env = make_training_env(case.task_config)
                env.reset(seed=s)
                pr = np.array(env._current_snapshot.robot_positions)
                pt = np.array(env._current_snapshot.target_positions)
                env.close()
                d = np.linalg.norm(pr[:, None, :] - pt[None, :, :], axis=2)
                assign = tuple(int(x) for x in d.argmin(axis=1))
                R, _c, _k = run_episode(case, s, assign, mode)
                vals.append(R / HORIZON)
            js[name] = float(np.mean(vals))
        S = 500.0 * (js["basic(uniform)"] + js["coop(crossing)"])
        print(f"  mode={mode:6s} J_basic={js['basic(uniform)']:.4f} J_coop={js['coop(crossing)']:.4f} "
              f" ==> S ≈ {S:6.1f}")


if __name__ == "__main__":
    main()
