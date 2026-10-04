"""P666 规则策略 V1：分量饱和驱动 + 预测末态分工 + 轻量避碰 + 记忆外推。

设计依据见开发笔记中的作战方案（PLAN.md）第 3 节（第一性分析）与第 5 节（提分清单）：

- 一局 10 步、dt = 0.1 s，动作箱 [-1, 1]^2 使 |u| 最大到 sqrt(2)：轴向满推 10 步位移 0.249 m，
  对角满推 0.3522 m（+41%）。故主控制律用分量饱和 u = clip(K * rel, -1, 1)，而非单位向量。
- 覆盖判定半径 0.15 m、感知半径 0.6 m；目标 10 步只移动 0.07-0.14 m。
- 只使用 AgentObservation 的合法字段，不访问环境内部状态、全局状态或队友私有状态。
- 策略内部自带动作保险链（nan_to_num -> clip -> float32 连续数组），返回 (2,) 的 float32。
- 依赖：numpy + coverage_bench.protocol 的类型 + 标准库（json/pathlib/typing）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    # ---- E008（行为变更轮）：诊断驱动，已实测通过公开硬门（S 221.667 → 225.000）----
    # 依据（诊断实测）：coop-0 的碰撞是 step 9 的 r0–r1 正面对冲（cosθ≈−0.99），
    # 碰撞前一步 d=0.1481 > safety_radius=0.12 ⇒ 原径向排斥未激活；且 g=4/视界 0.2 时
    # 虽触发但动量已积累、动作被裁剪回原角落 ⇒ 无效。故需"看得更远 + 让行强度压过饱和追踪"。
    "pred_avoid": True,         # E008-a 预测性避撞（默认开启，值来自消融实测最优档）
    "t_lookahead": 0.4,         # 预测接触时间视界（秒）；第 3 轮实测 0.4 是唯一"块 A 零退化"档
    "tangential_bias": 0.7,     # 对冲时切向权重（1.0 被裁剪失效、0.0 丢覆盖率，0.7 通过）
    "contact_margin": 0.015,    # 接触判定余量（米）；r_contact = 2*robot_radius + margin
    "avoid_gain": 16.0,         # 让行强度（8 不足、16 过闸）
    "deadzone_radius": 0.0,     # E008-b 覆盖圈死区：公开与 16 块口径实测均无影响，故默认关闭
    "dz_gain": 4.0,             # 死区内增益（保留可测开关）
}

_CONFIG_FILENAME = "rule-config.json"

_PREDICT_HORIZON_CHOICES = ("remaining", "zero")
_BLIND_ACTION_CHOICES = ("center", "stay")

# 逐键的合法取值域（闭区间；越界即视为非法，回退默认值）。
# 下界取 0 是刻意的：safety_radius=0.0 表示关闭避碰、control_gain=0.0 表示不驱动，
# 都是消融实验会用到的合法配置，不能被清洗掉。
_GAIN_RANGE = (0.0, 1.0e3)              # control_gain
_SAFETY_RADIUS_RANGE = (0.0, 5.0)       # safety_radius: 0.0 关闭避碰
_SAFETY_GAIN_RANGE = (0.0, 1.0e3)       # safety_gain: 0.0 排斥强度为零
_CLIP_EPS_RANGE = (0.0, 1.0)            # clip_eps: 仅作数值下限
_T_LOOKAHEAD_RANGE = (0.1, 1.0)         # t_lookahead（诊断：0.2 太短，动量已积累）
_TANGENTIAL_BIAS_RANGE = (0.0, 1.0)     # tangential_bias
_CONTACT_MARGIN_RANGE = (0.0, 0.05)     # contact_margin
_AVOID_GAIN_RANGE = (0.0, 20.0)         # avoid_gain
_DEADZONE_RADIUS_RANGE = (0.0, 0.2)     # deadzone_radius: 0.0 关闭死区
_DZ_GAIN_RANGE = (0.0, 10.0)            # dz_gain

#: 数值键 -> 取值域。**用 .get() 兜底**：将来若新增数值键而忘记登记，
#: 最坏也只是"该键不做范围校验"，绝不抛 KeyError —— 因为 KeyError 会逃出
#: `load_config`（其 try 只覆盖读文件与 JSON 解析），进而让 build_policy 抛异常，
#: 使该局变成 protocol/runtime 错误、整次评测 performance_score = null。
_RANGE_BY_KEY: Dict[str, Tuple[float, float]] = {
    "control_gain": _GAIN_RANGE,
    "safety_radius": _SAFETY_RADIUS_RANGE,
    "safety_gain": _SAFETY_GAIN_RANGE,
    "clip_eps": _CLIP_EPS_RANGE,
    "t_lookahead": _T_LOOKAHEAD_RANGE,
    "tangential_bias": _TANGENTIAL_BIAS_RANGE,
    "contact_margin": _CONTACT_MARGIN_RANGE,
    "avoid_gain": _AVOID_GAIN_RANGE,
    "deadzone_radius": _DEADZONE_RADIUS_RANGE,
    "dz_gain": _DZ_GAIN_RANGE,
}

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
        if key in ("unit_magnitude", "deconflict", "memory", "pred_avoid"):
            if isinstance(value, bool):
                merged[key] = value
            continue

        number = _finite_float(value)
        if number is None:
            continue
        bounds = _RANGE_BY_KEY.get(key)
        if bounds is None:
            # 未登记取值域的数值键：不做范围校验直接采纳（.get 兜底，绝不 KeyError）。
            merged[key] = number
            continue
        low, high = bounds
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


def _time_to_contact(rel_pos: np.ndarray, rel_vel: np.ndarray, r_contact: float) -> Optional[float]:
    """E008-a：最早接触时间 t≥0（|rel_pos + rel_vel·t| ≤ r_contact 的最小解）；无解返回 None。

    相对运动为匀速外推（协议给了队友相对速度 peers[k,2:4]，直接用，不做差分估计）。
    """
    a = float(rel_vel @ rel_vel)
    c = float(rel_pos @ rel_pos) - r_contact * r_contact
    if c <= 0.0:
        return 0.0                      # 已处于接触半径内
    if a <= 1e-12:
        return None                     # 相对静止且在外侧：永不接触
    b = 2.0 * float(rel_pos @ rel_vel)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = (-b - math.sqrt(disc)) / (2.0 * a)
    if root < 0.0:
        return None                     # 最近点在过去（正在远离）
    return float(root)


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
        # E008 需要本局公开物理参数（全部来自 EpisodeContext.task，不硬编码）
        self._robot_radius = 0.05
        self._target_radius = 0.15
        # ---- 只读诊断计数器（F3/F11/E008）：只做整数记账，不参与决策、不改变动作 ----
        # 由 tools_local/ 的本地工具读取与断言；act/reset 内绝不抛异常。
        self.stats: Dict[str, int] = {
            "act_calls": 0,
            "error_count": 0,
            "no_visible_target_steps": 0,
            "memory_hits": 0,
            "blind_center_steps": 0,
            "stop_now_steps": 0,
            "pred_avoid_steps": 0,
            "deadzone_steps": 0,
        }

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
            rr = _finite_float(getattr(task, "robot_radius", None))
            tr = _finite_float(getattr(task, "target_radius", None))
            if rr is not None and rr > 0.0:
                self._robot_radius = rr
            if tr is not None and tr > 0.0:
                self._target_radius = tr
        except Exception:
            self._agent_index = 0
            self._num_agents = 0
            self._num_targets = 0
            self._horizon = 0
            self._dt = 0.0
            self._robot_radius = 0.05
            self._target_radius = 0.15
        finally:
            self._history = {}
            self._chosen = None
            self._last_step_index = -1
            self._error_count = 0
            for _key in self.stats:
                self.stats[_key] = 0

    def close(self) -> None:
        """无资源需要释放。"""
        return None

    # ------------------------------------------------------------------ 每步决策
    def act(self, observation):
        """返回 (2,) float32 动作，范围 [-1, 1]；内部异常时退回零动作（有效性保险）。"""
        self.stats["act_calls"] += 1
        try:
            return self._act_impl(observation)
        except Exception:
            self._error_count += 1
            self.stats["error_count"] = self._error_count
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
            # 诊断计数：本步没有任何可见目标可参与选择（不改变决策）。
            self.stats["no_visible_target_steps"] += 1

        # ---- 3) 生成控制目标：当前可见用现测相对位置；不可见且开启记忆则外推 ----
        visible_set = set(visible_now)
        goal_rel = None
        goal_vel = np.zeros(2)
        stop_now = False
        if chosen is not None:
            if chosen in visible_set:
                goal_rel = self._history[chosen][1]
                goal_vel = self._history[chosen][2]
            elif cfg["memory"] and chosen in self._history:
                # 诊断计数：已选目标当前不可见、走记忆外推分支（F3 关注的路径）。
                self.stats["memory_hits"] += 1
                last_step, last_rel, last_vel = self._history[chosen]
                elapsed = max(step_index - last_step, 0)
                extrapolated = last_rel + last_vel * dt * float(elapsed)
                if float(np.linalg.norm(extrapolated)) <= cfg["clip_eps"]:
                    # 外推距离 <= 0（已贴在目标上）：停止追逐，不再生成驱动。
                    stop_now = True
                else:
                    goal_rel = extrapolated
                    goal_vel = last_vel

        if stop_now:
            self.stats["stop_now_steps"] += 1
            action = np.zeros(2, dtype=np.float64)
        elif goal_rel is not None:
            # E008-b：死区只作用于"追目标"路径（is_target=True）。
            action = self._control(goal_rel, goal_vel, True)
        elif cfg["blind_action"] == "center":
            # 无可见目标：朝场地中心（self_state[0:2] 即自身绝对位置）驱动。
            # is_target=False ⇒ **盲走分支不应用死区**，行为与 E001–E007 逐位相同（§2.2）。
            self.stats["blind_center_steps"] += 1
            action = self._control(-self_pos, None, False)
        else:
            self.stats["blind_center_steps"] += 1
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

        # ---- 4b) E008-a 预测性避撞（pred_avoid，默认关闭）----
        # 动机（诊断实测）：coop-0 的碰撞是 r0–r1 正面对冲（cosθ≈−0.99），碰撞前一步
        # d=0.1481 > safety_radius=0.12 ⇒ 现有径向排斥**未激活**、修正量为 0；而阈值放大到
        # 0.30 会改变动作 ⇒ 病因是"阈值太晚"，不是力度不足。故改用"预测接触时间"提前让行。
        if bool(cfg["pred_avoid"]):
            t_look = float(cfg["t_lookahead"])
            r_contact = 2.0 * float(self._robot_radius) + float(cfg["contact_margin"])
            tan_bias = float(cfg["tangential_bias"])
            avoid_gain = float(cfg["avoid_gain"])
            for k in range(min(len(peer_visible), peers.shape[0])):
                if k == agent_index or not peer_visible[k]:
                    continue
                rel_pos = np.asarray(peers[k, :2], dtype=np.float64)
                rel_vel = np.asarray(peers[k, 2:4], dtype=np.float64)
                t_hit = _time_to_contact(rel_pos, rel_vel, r_contact)
                if t_hit is None or t_hit > t_look:
                    continue
                self.stats["pred_avoid_steps"] += 1
                d = float(np.linalg.norm(rel_pos))
                speed = float(np.linalg.norm(rel_vel))
                denom = max(d, clip_eps, 1e-12)
                radial = -rel_pos / denom
                dir_vec = radial
                if d > clip_eps and speed > clip_eps:
                    cos_t = float(rel_pos @ rel_vel) / (d * speed)
                    if cos_t < -0.7:
                        # 对冲：取切向让行。**符号必须成对反对称**（按编号大小），
                        # 否则两台车会同侧让行而继续对撞。
                        tangent = np.array([-rel_pos[1], rel_pos[0]], dtype=np.float64)
                        tn = float(np.linalg.norm(tangent))
                        if tn > clip_eps:
                            sign = 1.0 if agent_index < k else -1.0
                            dir_vec = tan_bias * (sign * tangent / tn) + (1.0 - tan_bias) * radial
                gain_t = avoid_gain * max(0.0, 1.0 - (t_hit / t_look)) if t_look > 0.0 else avoid_gain
                action = action + gain_t * dir_vec

        # ---- 5) 动作保险链：nan/inf 清洗 -> 裁剪 -> float32 连续数组，shape (2,) ----
        safe = np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        safe = np.clip(safe, -1.0, 1.0)
        safe = np.ascontiguousarray(safe, dtype=np.float32)
        return safe.reshape(2)

    # ------------------------------------------------------------------ 内部工具
    def _control(self, rel, rel_vel=None, is_target: bool = True) -> np.ndarray:
        """控制律：unit_magnitude 时用单位向量（E001 对照组），否则分量饱和驱动。

        `is_target=False` 时**不应用覆盖圈死区**（盲走分支必须逐位不变，见 §2.2）。
        """
        rel = np.asarray(rel, dtype=np.float64).reshape(-1)[:2]
        distance = float(np.linalg.norm(rel))
        if distance <= float(self.config["clip_eps"]):
            return np.zeros(2, dtype=np.float64)
        if self.config["unit_magnitude"]:
            return rel / distance
        gain = float(self.config["control_gain"])
        if is_target:
            gain = self._effective_gain(rel, distance, rel_vel)
        return gain * rel

    def _effective_gain(self, rel: np.ndarray, distance: float, rel_vel) -> float:
        """E008-b 覆盖圈死区：进圈后改用小增益，抑制过冲/挤压。

        仅当 **①距离 ≤ deadzone_radius** 且 **②预测下一步仍在覆盖圈内**（条件②防止目标外逃时丢覆盖）
        时改用 `dz_gain`；否则维持 `control_gain`。`target_radius` 取自 EpisodeContext.task，不硬编码。
        """
        cfg = self.config
        base = float(cfg["control_gain"])
        dz = float(cfg["deadzone_radius"])
        if dz <= 0.0 or distance > dz:
            return base
        if self._target_radius <= 0.0 or self._dt <= 0.0:
            return base
        rv = np.zeros(2) if rel_vel is None else np.asarray(rel_vel, dtype=np.float64).reshape(-1)[:2]
        predicted = rel + rv * self._dt
        if float(np.linalg.norm(predicted)) <= self._target_radius:
            self.stats["deadzone_steps"] += 1
            return float(cfg["dz_gain"])
        return base

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
