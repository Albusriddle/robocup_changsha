r"""官方环境实测探针（Python 3.12 + mpe2==1.1.1）。

目的：把方案里所有关于"机器人 10 步能走多远"的假设换成实测值。
关键点：mpe2 1.1.1 的 World.integrate_state 是**先位置后速度**：
    p += v * dt ;  v *= (1 - damping) ;  v += (F/m) * dt ;  |v| <= max_speed
因此第 1 步位移为 0（初速为 0），与经典 MPE 的相位不同，可达距离更小。

用法（仓库根目录，使用评测环境）：
    .venv\Scripts\python.exe participant\P666\tools_local\real_dynamics_probe.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # PYC_GUARD：不写出 __pycache__（官方预检把被忽略文件也算未提交内容）
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # 仓库根: tools_local -> P666 -> participant -> <repo>
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402


def build_env(layout_index: int = 0, seed: int = 1001):
    suite = load_suite(REPO / "configs" / "public-suite-v1.yaml")
    case = suite.groups[0].cases[layout_index]
    env = make_training_env(case.task_config)
    env.reset(seed=seed)
    return env, case


def displacement_profile(action_value: float, steps: int = 10) -> None:
    env, case = build_env()
    agents = list(env.agents)
    start = np.array(env._current_snapshot.robot_positions, dtype=float).copy()
    traj = []
    for k in range(steps):
        if not env.agents:
            break
        acts = {}
        for i, a in enumerate(agents):
            acts[a] = np.array([action_value, 0.0], dtype=np.float32) if i == 0 else np.zeros(2, dtype=np.float32)
        env.step(acts)
        pos = np.array(env._current_snapshot.robot_positions, dtype=float)
        vel = np.array(env._current_snapshot.robot_velocities, dtype=float)
        traj.append((k + 1, pos[0].copy(), vel[0].copy(), float(np.linalg.norm(pos[0] - start[0]))))
    print(f"\n--- 恒定动作 u=({action_value}, 0)，机器人 0 的实测轨迹（seed={1001}）---")
    for k, pos, vel, disp in traj:
        print(f"  step {k:2d}: pos=({pos[0]:+.4f},{pos[1]:+.4f})  v=({vel[0]:+.4f},{vel[1]:+.4f}) "
              f"|v|={np.linalg.norm(vel):.4f}  累计位移={disp:.4f}")
    print(f"  10 步累计位移 = {traj[9][3]:.4f} m  (方案假设值 0.287 / 修正后 0.249)")


def target_profile(steps: int = 10) -> None:
    env, case = build_env()
    agents = list(env.agents)
    t0 = np.array(env._current_snapshot.target_positions, dtype=float).copy()
    print("\n--- 目标运动实测 ---")
    print("  初速:", np.round(np.linalg.norm(env._current_snapshot.target_velocities, axis=1), 4).tolist())
    for k in range(steps):
        acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
        env.step(acts)
        t = np.array(env._current_snapshot.target_positions, dtype=float)
        d = np.linalg.norm(t - t0, axis=1)
        print(f"  step {k+1:2d}: |Δ|={np.round(d, 4).tolist()}")


def reward_check() -> None:
    """验证奖励 = C_t - 0.2*K_t，且 C_t 是二部图最大匹配。"""
    env, case = build_env()
    agents = list(env.agents)
    acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
    obs, rew, term, trunc, infos = env.step(acts)
    a0 = agents[0]
    print("\n--- 奖励结构实测（全零动作，第 1 步）---")
    print("  reward_terms:", infos[a0]["reward_terms"])
    print("  metrics:", {k: v for k, v in infos[a0]["metrics"].items()} if isinstance(infos[a0]["metrics"], dict) else infos[a0]["metrics"])
    print("  team reward == coverage - 0.2*collision :",
          abs(rew[a0] - (infos[a0]["reward_terms"]["coverage"] - 0.2 * infos[a0]["reward_terms"]["collision"])) < 1e-12)


def collision_probe() -> None:
    """构造两机器人近距离场景，测量接触力对速度的影响。"""
    env, case = build_env()
    st = env._scenario_state
    st.robot_positions[0] = np.array([0.0, 0.0])
    st.robot_positions[1] = np.array([0.12, 0.0])
    st.robot_positions[2] = np.array([0.8, 0.8])
    for i in range(3):
        st.mpe_world.agents[i].state.p_pos = np.array(st.robot_positions[i], dtype=float)
        st.mpe_world.agents[i].state.p_vel = np.zeros(2, dtype=float)
    agents = list(env.agents)
    acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
    obs, rew, term, trunc, infos = env.step(acts)
    print("\n--- 接触力实测（机器人 0/1 初始中心距 0.12，半径和 0.10）---")
    print("  步后位置:", np.round(np.array(env._current_snapshot.robot_positions), 4).tolist())
    print("  步后速度:", np.round(np.array(env._current_snapshot.robot_velocities), 4).tolist())
    print("  metrics:", infos[agents[0]]["metrics"])
    print("  reward:", rew[agents[0]])

    for gap in (0.10, 0.09, 0.06, 0.02):
        env2, _ = build_env()
        st2 = env2._scenario_state
        st2.robot_positions[0] = np.array([0.0, 0.0])
        st2.robot_positions[1] = np.array([gap, 0.0])
        st2.robot_positions[2] = np.array([0.9, 0.9])
        for i in range(3):
            st2.mpe_world.agents[i].state.p_pos = np.array(st2.robot_positions[i], dtype=float)
            st2.mpe_world.agents[i].state.p_vel = np.zeros(2, dtype=float)
        acts2 = {a: np.zeros(2, dtype=np.float32) for a in list(env2.agents)}
        env2.step(acts2)
        v0 = np.array(env2._current_snapshot.robot_velocities[0])
        print(f"  中心距 {gap:.2f} -> 机器人0 速度 {np.round(v0, 4).tolist()} "
              f"(|v|={np.linalg.norm(v0):.4f})  metrics={env2._current_snapshot and ''}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    print("=" * 78)
    print("官方环境实测（mpe2 1.1.1, Python 3.12）")
    print("=" * 78)
    displacement_profile(1.0)
    displacement_profile(0.5)
    target_profile()
    reward_check()
    collision_probe()
