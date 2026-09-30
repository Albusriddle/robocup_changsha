"""P666 规则策略 V1：分量饱和驱动 + 预测末态分工 + 轻量避碰 + 记忆外推。

设计依据见 F:\\DSHworkspace\\robocup_changsha-notes\\PLAN.md 第 3 节（第一性分析）与第 5 节（提分清单）：

- 一局 10 步、dt = 0.1 s，动作箱 [-1, 1]^2 使 |u| 最大到 sqrt(2)：轴向满推 10 步位移 0.249 m，
  对角满推 0.3522 m（+41%）。故主控制律用分量饱和 u = clip(K * rel, -1, 1)，而非单位向量。
- 覆盖判定半径 0.15 m、感知半径 0.6 m；目标 10 步只移动 0.07-0.14 m。
- 只使用 AgentObservation 的合法字段，不访问环境内部状态、全局状态或队友私有状态。
- 策略内部自带动作保险链（nan_to_num -> clip -> float32 连续数组），返回 (2,) 的 float32。
- 依赖：numpy + coverage_bench.protocol 的类型 + 标准库（json/pathlib/typing）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "DEFAULT_CONFIG",
    "CoverageRulePolicy",
    "build_policy",
    "load_config",
]

#: 冻结的超参数默认值（键名与语义即对外契约，不得改名）。
DEFAULT_CONFIG: Dict[str, Any] = {
    "control_gain": 20.0,
    "predict_horizon": "remaining",
    "unit_magnitude": False,
    "deconflict": True,
    "safety_radius": 0.12,
    "safety_gain": 4.0,
    "memory": True,
    "blind_action": "center",
    "clip_eps": 1e-6,
}

_CONFIG_FILENAME = "rule-config.json"

_PREDICT_HORIZON_CHOICES = ("remaining", "zero")
_BLIND_ACTION_CHOICES = ("center", "stay")

# 逐键的合法取值域（闭区间；越界即视为非法，回退默认值）。
# 下界取 0 是刻意的：safety_radius=0.0 表示关闭避碰、control_gain=0.0 表示不驱动，
# 都是消融实验会用到的合法配置，不能被清洗掉。
_GAIN_RANGE = (0.0, 1.0e3)          # control_gain
_SAFETY_RADIUS_RANGE = (0.0, 5.0)   # safety_radius: 0.0 关闭避碰
_SAFETY_GAIN_RANGE = (0.0, 1.0e3)   # safety_gain: 0.0 排斥强度为零
_CLIP_EPS_RANGE = (0.0, 1.0)        # clip_eps: 仅作数值下限

#: 目标选择里"等距"的数值判等容差（浮点比较用）。
_TIE_EPS = 1e-9


def _finite_float(value: Any) -> Optional[float]:
    """把数值（int/float/numpy 标量）转成有限 float；bool、字符串、非有限值返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(out):
            return None
        return out
    return None


def _merge_config(raw: Any) -> Dict[str, Any]:
    """以 DEFAULT_CONFIG 为底，只合并已知键的合法取值；未知键忽略，非法值回退默认。

    本函数保证对任意输入都不抛异常，且返回一份全新的 dict（不共享 DEFAULT_CONFIG 内部对象）。
    """
    merged: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return merged

    for key in DEFAULT_CONFIG:
        if key not in raw:
            continue
        value = raw[key]

        if key == "predict_horizon":
            if isinstance(value, str) and value in _PREDICT_HORIZON_CHOICES:
                merged[key] = value
            continue
        if key == "blind_action":
            if isinstance(value, str) and value in _BLIND_ACTION_CHOICES:
                merged[key] = value
            continue
        if key in ("unit_magnitude", "deconflict", "memory"):
            if isinstance(value, bool):
                merged[key] = value
            continue

        number = _finite_float(value)
        if number is None:
            continue
        low, high = {
            "control_gain": _GAIN_RANGE,
            "safety_radius": _SAFETY_RADIUS_RANGE,
            "safety_gain": _SAFETY_GAIN_RANGE,
            "clip_eps": _CLIP_EPS_RANGE,
        }[key]
        if low <= number <= high:
            merged[key] = number
    return merged


def load_config(artifact_dir) -> Dict[str, Any]:
    """读取 artifact_dir/"rule-config.json" 并清洗；任何异常都回退 DEFAULT_CONFIG。"""
    try:
        config_path = Path(artifact_dir) / _CONFIG_FILENAME
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        # 目录缺失、文件不存在、编码/JSON 损坏、路径类型非法 —— 一律用默认配置。
        return dict(DEFAULT_CONFIG)
    return _merge_config(raw)


class CoverageRulePolicy:
    """纯规则策略：分量饱和追踪 + 预测末态目标选择 + 轻量避碰 + 记忆外推。

    构造路径：
        CoverageRulePolicy(config=None, artifact_dir=None)   # config 为 None 时等价 DEFAULT_CONFIG
        build_policy(context)                                # 等价 CoverageRulePolicy(load_config(...), ...)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, artifact_dir=None):
        self.config: Dict[str, Any] = _merge_config(config) if config is not None else dict(DEFAULT_CONFIG)
        self.artifact_dir = artifact_dir

        # ---- 每局状态（reset 必须全部清空）----
        self._agent_index = 0
        self._num_agents = 0
        self._num_targets = 0
        self._horizon = 0
        self._dt = 0.0
        self._history: Dict[int, Any] = {}
        self._chosen: Optional[int] = None
        self._last_step_index = -1
        self._error_count = 0

    # ------------------------------------------------------------------ 生命周期
    def reset(self, context) -> None:
        """开始新一局：读取回合上下文并清空全部每局状态。"""
        try:
            self._agent_index = int(getattr(context, "agent_index", 0))
            self._num_agents = int(getattr(context, "num_agents", 0))
            self._num_targets = int(getattr(context, "num_targets", 0))
            self._horizon = int(getattr(context, "horizon", 0))
            task = getattr(context, "task", None)
            dt = _finite_float(getattr(task, "dt", None))
            if dt is None or dt <= 0.0:
                dt = _finite_float(getattr(context, "dt", None))
            self._dt = dt if (dt is not None and dt > 0.0) else 0.0
        except Exception:
            self._agent_index = 0
            self._num_agents = 0
            self._num_targets = 0
            self._horizon = 0
            self._dt = 0.0
        finally:
            self._history = {}
            self._chosen = None
            self._last_step_index = -1
            self._error_count = 0

    def close(self) -> None:
        """无资源需要释放。"""
        return None

    # ------------------------------------------------------------------ 每步决策
    def act(self, observation):
        """返回 (2,) float32 动作，范围 [-1, 1]；内部异常时退回零动作（有效性保险）。"""
        try:
            return self._act_impl(observation)
        except Exception:
            self._error_count += 1
            return np.zeros(2, dtype=np.float32)

    def _act_impl(self, observation):
        cfg = self.config
        dt = self._dt

        self_pos = np.asarray(observation["self_state"], dtype=np.float64).reshape(-1)[:2].copy()
        targets = np.asarray(observation["targets"], dtype=np.float64).reshape(-1, 3)
        target_exists = np.asarray(observation["target_exists"]).reshape(-1).astype(bool)
        target_visible = np.asarray(observation["target_visible"]).reshape(-1).astype(bool)
        peers = np.asarray(observation["peers"], dtype=np.float64).reshape(-1, 5)
        peer_visible = np.asarray(observation["peer_visible"]).reshape(-1).astype(bool)

        step_index = int(observation["step_index"])
        agent_index = int(observation["agent_index"])
        self._last_step_index = step_index

        # ---- 1) 只用可见且存在的目标；用相邻步相对位置差分估计相对速度 ----
        visible_now = []
        count = min(len(targets), len(target_exists), len(target_visible))
        for j in range(count):
            if not (target_visible[j] and target_exists[j]):
                continue
            visible_now.append(j)
            rel = targets[j, :2].copy()
            previous = self._history.get(j)
            if previous is None:
                rel_vel = np.zeros(2)
            elif previous[0] == step_index - 1 and dt > 0.0:
                rel_vel = (rel - previous[1]) / dt
            else:
                # 目标上一拍不可见（或 dt 非法）：不跨步差分，沿用上次估计值。
                rel_vel = np.array(previous[2], dtype=np.float64, copy=True)
            self._history[j] = (step_index, rel, rel_vel)

        # ---- 2) 目标选择：预测末态相对距离升序，deconflict 时让位给更近的可见队友 ----
        steps_left = self._horizon - step_index
        ranked = []
        for j in visible_now:
            _s, rel, rel_vel = self._history[j]
            distance = float(np.linalg.norm(rel))
            if cfg["predict_horizon"] == "remaining" and steps_left > 0 and dt > 0.0:
                key = float(np.linalg.norm(rel + rel_vel * dt * float(steps_left)))
            else:
                key = distance
            ranked.append((key, distance, j, rel))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))

        if ranked:
            chosen = ranked[0][2]
            if cfg["deconflict"]:
                for _key, distance, j, rel in ranked:
                    if not self._blocked_by_peer(j, rel, distance, agent_index, peers, peer_visible):
                        chosen = j
                        break
            self._chosen = chosen
        else:
            chosen = self._chosen

        # ---- 3) 生成控制目标：当前可见用现测相对位置；不可见且开启记忆则外推 ----
        visible_set = set(visible_now)
        goal_rel = None
        stop_now = False
        if chosen is not None:
            if chosen in visible_set:
                goal_rel = self._history[chosen][1]
            elif cfg["memory"] and chosen in self._history:
                last_step, last_rel, last_vel = self._history[chosen]
                elapsed = max(step_index - last_step, 0)
                extrapolated = last_rel + last_vel * dt * float(elapsed)
                if float(np.linalg.norm(extrapolated)) <= cfg["clip_eps"]:
                    # 外推距离 <= 0（已贴在目标上）：停止追逐，不再生成驱动。
                    stop_now = True
                else:
                    goal_rel = extrapolated

        if stop_now:
            action = np.zeros(2, dtype=np.float64)
        elif goal_rel is not None:
            action = self._control(goal_rel)
        elif cfg["blind_action"] == "center":
            # 无可见目标：朝场地中心（self_state[0:2] 即自身绝对位置）驱动。
            action = self._control(-self_pos)
        else:
            action = np.zeros(2, dtype=np.float64)

        # ---- 4) 轻量避碰：仅对可见、距离 < safety_radius 的队友施加排斥 ----
        safety_radius = float(cfg["safety_radius"])
        safety_gain = float(cfg["safety_gain"])
        clip_eps = float(cfg["clip_eps"])
        if safety_radius > 0.0:
            for k in range(min(len(peer_visible), peers.shape[0])):
                if k == agent_index or not peer_visible[k]:
                    continue
                peer_rel = peers[k, :2]
                distance = float(np.linalg.norm(peer_rel))
                if distance < safety_radius:
                    strength = (safety_radius - distance) / safety_radius * safety_gain
                    denom = max(distance, clip_eps, 1e-12)
                    # 排斥 = 沿"由队友指向自身"的方向：away = -peers[k][:2] / d。
                    # 注意：契约文字写成 `u -= f * (self_pos - peer_pos)/d`，而观测里只有
                    # 队友相对量 peers[k][:2] = peer_abs - self_abs，故 (self_pos - peer_pos)
                    # 恰等于 -peers[k][:2]；若按字面 `-=` 落地会变成"朝队友靠近"（吸引），
                    # 与"避碰"语义、PLAN 第 3.4 节（安全半径 0.30 m 掉分）及参考探针
                    # strategy_sweep.SatPolicy（`u -= f * prel/d`，prel = peer - self）相反。
                    # 这里按"排斥"落地；若要改回字面写法，只需把下面一行改成减号。
                    away = -peer_rel / denom
                    action = action + strength * away

        # ---- 5) 动作保险链：nan/inf 清洗 -> 裁剪 -> float32 连续数组，shape (2,) ----
        safe = np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        safe = np.clip(safe, -1.0, 1.0)
        safe = np.ascontiguousarray(safe, dtype=np.float32)
        return safe.reshape(2)

    # ------------------------------------------------------------------ 内部工具
    def _control(self, rel) -> np.ndarray:
        """控制律：unit_magnitude 时用单位向量（E001 对照组），否则分量饱和驱动。"""
        rel = np.asarray(rel, dtype=np.float64).reshape(-1)[:2]
        distance = float(np.linalg.norm(rel))
        if distance <= float(self.config["clip_eps"]):
            return np.zeros(2, dtype=np.float64)
        if self.config["unit_magnitude"]:
            return rel / distance
        return float(self.config["control_gain"]) * rel

    def _blocked_by_peer(self, target_index, rel, distance, agent_index, peers, peer_visible) -> bool:
        """该目标是否已被"更近的可见队友"占用（更近，或等距且编号更小）。"""
        for k in range(min(len(peer_visible), peers.shape[0])):
            if k == agent_index or not peer_visible[k]:
                continue
            peer_rel = peers[k, :2]
            peer_to_target = float(np.linalg.norm(peer_rel - rel))
            if peer_to_target < distance - _TIE_EPS:
                return True
            if abs(peer_to_target - distance) <= _TIE_EPS and k < agent_index:
                return True
        return False


def build_policy(context) -> CoverageRulePolicy:
    """官方入口：等价于 CoverageRulePolicy(load_config(context.artifact_dir), context.artifact_dir)。"""
    return CoverageRulePolicy(load_config(context.artifact_dir), context.artifact_dir)
