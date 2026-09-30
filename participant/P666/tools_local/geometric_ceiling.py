"""几何上界：如果策略"提前知道整局目标轨迹"，最多能拿多少分？

模型（与实测动力学一致）：
  - 位置先于速度：p += v*dt ; v = 0.75*v + 0.1*u ; |v| <= 1
  - 动作箱 [-1,1]^2 => u = clip(sqrt(2) * unit(theta))，|u| 最大 sqrt(2)
  - 目标轨迹与机器人动作无关，因此可先用零动作跑一局拿到真值轨迹
  - 每台机器人在 10 步内只够走一条近似直线（总位移 0.35 m），
    所以"选一个朝向 theta，全程满推"就是一个非常强的可达上界
  - 覆盖按官方最大匹配逐步统计；theta 用候选集 + 枚举联合优化

结论用途：判断"性能分还有多少空间"决定投入方向（规则精修 / 强化学习 / 实验与工程分）。

用法（仓库根目录，评测环境）：
    .venv\\Scripts\\python.exe participant\\P666\\tools_local\\geometric_ceiling.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO = Path(__file__).resolve().parents[3]  # 仓库根: tools_local -> P666 -> participant -> <repo>
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

HORIZON = 10
N_AGENTS = 3
N_TARGETS = 3
R_TARGET = 0.15
DT = 0.1
DAMP = 0.25
HALF_EXTENT = 1.0
R_ROBOT = 0.05
N_HEADINGS = 72
TOP_K = 6


def record_targets(case, seed: int):
    """零动作跑一局，拿到机器人初始位置与逐步目标位置（目标运动与动作无关）。"""
    env = make_training_env(case.task_config)
    env.reset(seed=seed)
    agents = list(env.agents)
    p0 = np.array(env._current_snapshot.robot_positions, dtype=float).copy()
    traj = []
    for _ in range(HORIZON):
        acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
        env.step(acts)
        traj.append(np.array(env._current_snapshot.target_positions, dtype=float).copy())
    env.close()
    return p0, np.stack(traj)          # (10, M, 2)


def robot_path(p0: np.ndarray, theta: float) -> np.ndarray:
    """满推朝向 theta 的机器人轨迹，返回 (10,2)：第 t 步结束后的位置。"""
    u = np.clip(np.sqrt(2.0) * np.array([np.cos(theta), np.sin(theta)]), -1.0, 1.0)
    p = np.array(p0, dtype=float)
    v = np.zeros(2)
    out = []
    for _ in range(HORIZON):
        p = p + v * DT
        v = (1.0 - DAMP) * v + u * DT
        sp = float(np.linalg.norm(v))
        if sp > 1.0:
            v = v * (1.0 / sp)
        limit = HALF_EXTENT - R_ROBOT
        for ax in (0, 1):
            if p[ax] > limit:
                p[ax] = limit
                if v[ax] > 0:
                    v[ax] = 0.0
            elif p[ax] < -limit:
                p[ax] = -limit
                if v[ax] < 0:
                    v[ax] = 0.0
        out.append(p.copy())
    return np.stack(out)


def covered_steps_for_combo(paths: list[np.ndarray], tgt: np.ndarray) -> float:
    """给定三台机器人各自的轨迹，按官方最大匹配统计总覆盖步数。"""
    total = 0.0
    for t in range(HORIZON):
        adj = np.zeros((N_AGENTS, N_TARGETS), dtype=bool)
        for i in range(N_AGENTS):
            for j in range(N_TARGETS):
                if np.linalg.norm(paths[i][t] - tgt[t, j]) <= R_TARGET:
                    adj[i, j] = True
        matched = maximum_bipartite_matching(csr_matrix(adj))
        total += float(np.sum(matched >= 0))
    return total


def ceiling_single(case, seed: int) -> float:
    p0, tgt = record_targets(case, seed)
    thetas = np.linspace(0.0, 2.0 * np.pi, N_HEADINGS, endpoint=False)
    per_robot_paths = []
    for i in range(N_AGENTS):
        per_robot_paths.append([robot_path(p0[i], th) for th in thetas])
    # 每台机器人先各自挑出"能覆盖最多目标步数"的 TOP_K 个朝向
    cand = []
    for i in range(N_AGENTS):
        scores = []
        for k in range(N_HEADINGS):
            s = 0.0
            for j in range(N_TARGETS):
                for t in range(HORIZON):
                    if np.linalg.norm(per_robot_paths[i][k][t] - tgt[t, j]) <= R_TARGET:
                        s += 1.0
            scores.append(s)
        order = np.argsort(scores)[::-1][:TOP_K]
        cand.append([per_robot_paths[i][k] for k in order])
    best = 0.0
    for combo in itertools.product(range(len(cand[0])), range(len(cand[1])), range(len(cand[2]))):
        paths = [cand[0][combo[0]], cand[1][combo[1]], cand[2][combo[2]]]
        best = max(best, covered_steps_for_combo(paths, tgt))
    return best / (N_AGENTS * HORIZON)


def main() -> None:
    suite = load_suite(REPO / "configs" / "public-suite-v1.yaml")
    cases = [("basic(uniform)", suite.groups[0].cases[0], [1001, 1002] + list(range(5001, 5041))),
             ("coop(crossing)", suite.groups[1].cases[0], [2001, 2002] + list(range(6001, 6041)))]
    js = {}
    for name, case, seeds in cases:
        vals = [ceiling_single(case, s) for s in seeds]
        js[name] = float(np.mean(vals))
        print(f"  [{name}] 几何上界 J_max = {np.mean(vals):.4f} "
              f"(最大 {np.max(vals):.4f}, 最小 {np.min(vals):.4f}, {len(vals)} 例)")
    S = 500.0 * (js["basic(uniform)"] + js["coop(crossing)"])
    print(f"\n  ==> 几何上界 S_max ≈ {S:.1f}")
    print("  参照（同一批公开套件口径）：官方 P901 random 58.33 | P902 rule 175.00 | P903 PPO 116.67")
    print("  参照（30 随机种子的可实现规则策略，本目录 ride_control_probe）：S ≈ 121")


if __name__ == "__main__":
    main()
