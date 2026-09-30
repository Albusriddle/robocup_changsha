"""P666 私有开发种子集：固定常量 + 纯函数，不含任何随机抽样。

本种子集先固定、再使用、永不用于训练。

固定块（每块两个布局组，各 50 例）：
- 块 A：uniform 7001..7050（basic 组）、crossing 8001..8050（cooperation 组）
- 块 B：uniform 5001..5050、crossing 6001..6050（跨块复核用；与块 A 相互独立、不重叠）
- 公开块：1001 / 1002（uniform）、2001 / 2002（crossing），不计入任何私有块的 50 例

规则：种子列表一旦写入本文件就不再改动。改动会让历史实验不可比。
**块 A 是默认决策口径**（历史数字全部基于它，不得改动）；块 B 是跨块复现口径，
两块都给出同方向结论时才算"结论稳定"。绝对值只在同一块内可比，跨块不可比。

兼容性：`list_seeds("uniform", n)` / `list_seeds("crossing", n)` / `groups()` / `all_seeds()`
/ `describe()` 仍等价于块 A 的旧语义。
"""

from typing import Dict, Optional, Tuple
import sys

__all__ = [
    # 块 A（旧名，语义不变）
    "UNIFORM_SEEDS",
    "CROSSING_SEEDS",
    # 块 B
    "UNIFORM_SEEDS_B",
    "CROSSING_SEEDS_B",
    # 公开块
    "PUBLIC_UNIFORM_SEEDS",
    "PUBLIC_CROSSING_SEEDS",
    "PUBLIC_SEEDS",
    # 名字常量
    "GROUP_UNIFORM",
    "GROUP_CROSSING",
    "BLOCK_A",
    "BLOCK_B",
    "BLOCK_PUBLIC",
    "BLOCKS",
    # 纯函数
    "list_seeds",
    "groups",
    "blocks",
    "block_seeds",
    "public_seeds",
    "all_seeds",
    "group_of_public_seed",
    "describe",
    "describe_block",
]

# ---------------------------------------------------------------------------
# 固定常量（tuple，可哈希、不可原地修改）
# ---------------------------------------------------------------------------

GROUP_UNIFORM: str = "uniform"
GROUP_CROSSING: str = "crossing"

BLOCK_A: str = "A"
BLOCK_B: str = "B"
BLOCK_PUBLIC: str = "public"
BLOCKS: Tuple[str, ...] = (BLOCK_A, BLOCK_B, BLOCK_PUBLIC)

# --- 块 A（默认决策口径；常量与语义自 2026-09-30 首版起冻结）---
UNIFORM_SEEDS: Tuple[int, ...] = tuple(range(7001, 7051))   # 50 个
CROSSING_SEEDS: Tuple[int, ...] = tuple(range(8001, 8051))  # 50 个

# --- 块 B（跨块复现口径；独立常量，不覆盖块 A）---
UNIFORM_SEEDS_B: Tuple[int, ...] = tuple(range(5001, 5051))   # 50 个
CROSSING_SEEDS_B: Tuple[int, ...] = tuple(range(6001, 6051))  # 50 个

# --- 公开块（官方公开套件的 4 例）---
PUBLIC_UNIFORM_SEEDS: Tuple[int, ...] = (1001, 1002)
PUBLIC_CROSSING_SEEDS: Tuple[int, ...] = (2001, 2002)
PUBLIC_SEEDS: Tuple[int, ...] = PUBLIC_UNIFORM_SEEDS + PUBLIC_CROSSING_SEEDS

_BLOCK_GROUPS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    BLOCK_A: {GROUP_UNIFORM: UNIFORM_SEEDS, GROUP_CROSSING: CROSSING_SEEDS},
    BLOCK_B: {GROUP_UNIFORM: UNIFORM_SEEDS_B, GROUP_CROSSING: CROSSING_SEEDS_B},
    BLOCK_PUBLIC: {GROUP_UNIFORM: PUBLIC_UNIFORM_SEEDS, GROUP_CROSSING: PUBLIC_CROSSING_SEEDS},
}

# 旧版内部名（仍指向块 A，保持向后兼容）
_GROUPS: Dict[str, Tuple[int, ...]] = _BLOCK_GROUPS[BLOCK_A]

_PUBLIC_BY_GROUP: Dict[str, Tuple[int, ...]] = _BLOCK_GROUPS[BLOCK_PUBLIC]

_LEGACY_GROUP_NAMES: Tuple[str, ...] = (GROUP_UNIFORM, GROUP_CROSSING)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def _resolve_block(block: str) -> str:
    if block in _BLOCK_GROUPS:
        return block
    if block in _LEGACY_GROUP_NAMES:
        return BLOCK_A
    if block == "both":
        raise KeyError("块名不支持 'both'：请分别用 'A' 与 'B'（由调用方遍历）")
    raise KeyError(f"未知种子块: {block!r}，可用: {list(BLOCKS)}（或旧组名 {list(_LEGACY_GROUP_NAMES)}）")


def _head(seeds: Tuple[int, ...], n: Optional[int], what: str) -> Tuple[int, ...]:
    if n is None:
        return seeds
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"n 必须为正整数，收到: {n!r}")
    if n > len(seeds):
        raise ValueError(f"{what} 只有 {len(seeds)} 个固定种子，请求 {n} 个")
    return seeds[:n]


def list_seeds(block: str = BLOCK_A, group: Optional[str] = None, n: Optional[int] = None) -> Tuple[int, ...]:
    """返回固定种子（纯函数，无随机抽样）。

    用法:
        list_seeds("A")                    -> 块 A 全部（uniform 50 + crossing 50）
        list_seeds("B", "uniform", 10)     -> 块 B 的 uniform 前 10 个
        list_seeds("public")               -> 公开 4 例
        list_seeds("uniform", 3)           -> 旧语义：块 A 的 uniform 前 3 个（向后兼容）

    参数:
        block: "A" | "B" | "public"；也可传旧组名 "uniform"/"crossing"（等价块 A + 该组）。
        group: "uniform" | "crossing"；为 None 时返回整块（此时 n 必须为 None）。
        n: 按组取前 n 个（n=None 表示全部）。
    """
    if block in _LEGACY_GROUP_NAMES:
        # 旧调用 list_seeds("uniform", 3)：第二个位置参数是 n，不是 group
        if group is not None and not isinstance(group, str):
            if n is not None:
                raise TypeError("旧调用形式 list_seeds(group, n) 不接受第三个参数")
            n = group
            group = None
        if group is None:
            group = block
        block = BLOCK_A

    resolved = _resolve_block(block)
    block_map = _BLOCK_GROUPS[resolved]

    if group is None:
        if n is not None:
            raise ValueError("按整块取种子时不能同时指定 n；请给出 group 以按组取前 n 个")
        out: Tuple[int, ...] = ()
        for layout in (GROUP_UNIFORM, GROUP_CROSSING):
            out = out + block_map[layout]
        return out

    if group not in block_map:
        raise KeyError(f"未知布局组: {group!r}，可用: {list(block_map)}")
    return _head(block_map[group], n, f"块 {resolved} 的 {group}")


def groups(block: str = BLOCK_A) -> Dict[str, Tuple[int, ...]]:
    """返回 {布局组: 固定种子元组} 的副本（调用方修改不会影响本模块常量）。

    无参调用（旧语义）等价于 groups("A")。
    """
    return dict(_BLOCK_GROUPS[_resolve_block(block)])


def blocks() -> Tuple[str, ...]:
    """返回全部块名（"A", "B", "public"）。"""
    return BLOCKS


def block_seeds(block: str, n: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """按块返回 {布局组: 前 n 个种子} 的副本（跨块复现的便捷入口）。"""
    resolved = _resolve_block(block)
    return {layout: _head(seeds, n, f"块 {resolved} 的 {layout}")
            for layout, seeds in _BLOCK_GROUPS[resolved].items()}


def public_seeds() -> Tuple[int, ...]:
    """返回官方公开 4 例的 scenario_seed（顺序：uniform 两个、crossing 两个）。"""
    return PUBLIC_SEEDS


def all_seeds(block: str = BLOCK_A) -> Tuple[int, ...]:
    """返回某个块的全部种子（无参调用等价于块 A 的 100 个）。"""
    return list_seeds(block)


def group_of_public_seed(seed: int) -> str:
    """给出公开种子的布局组；未知种子抛 KeyError。"""
    for group, seeds in _PUBLIC_BY_GROUP.items():
        if seed in seeds:
            return group
    raise KeyError(f"{seed!r} 不是公开种子（{list(PUBLIC_SEEDS)}）")


def describe() -> str:
    """块 A + 公开块的旧版一行说明（保持向后兼容）。"""
    return (
        f"private: {GROUP_UNIFORM}={UNIFORM_SEEDS[0]}..{UNIFORM_SEEDS[-1]}"
        f"({len(UNIFORM_SEEDS)}), "
        f"{GROUP_CROSSING}={CROSSING_SEEDS[0]}..{CROSSING_SEEDS[-1]}"
        f"({len(CROSSING_SEEDS)}); public={list(PUBLIC_SEEDS)}"
    )


def describe_block(block: str) -> str:
    """按块生成一行说明（含块 B）。"""
    resolved = _resolve_block(block)
    if resolved == BLOCK_PUBLIC:
        return f"public: {GROUP_UNIFORM}={list(PUBLIC_UNIFORM_SEEDS)} {GROUP_CROSSING}={list(PUBLIC_CROSSING_SEEDS)}"
    block_map = _BLOCK_GROUPS[resolved]
    parts = []
    for layout in (GROUP_UNIFORM, GROUP_CROSSING):
        seeds = block_map[layout]
        parts.append(f"{layout}={seeds[0]}..{seeds[-1]}({len(seeds)})")
    return f"块 {resolved}: " + ", ".join(parts)


def _self_check() -> None:
    """轻量自检：块 A 冻结不变、块 B 独立、无重叠、旧调用兼容。"""
    # 块 A 冻结值
    assert UNIFORM_SEEDS == tuple(range(7001, 7051))
    assert CROSSING_SEEDS == tuple(range(8001, 8051))
    # 块 B
    assert len(UNIFORM_SEEDS_B) == 50 and len(CROSSING_SEEDS_B) == 50
    assert UNIFORM_SEEDS_B[0] == 5001 and UNIFORM_SEEDS_B[-1] == 5050
    assert CROSSING_SEEDS_B[0] == 6001 and CROSSING_SEEDS_B[-1] == 6050
    # 独立性
    assert not set(UNIFORM_SEEDS + CROSSING_SEEDS) & set(UNIFORM_SEEDS_B + CROSSING_SEEDS_B), "块 A 与块 B 不得重叠"
    assert not set(all_seeds(BLOCK_B)) & set(PUBLIC_SEEDS), "块 B 不得包含公开种子"
    assert not set(all_seeds(BLOCK_A)) & set(PUBLIC_SEEDS), "块 A 不得包含公开种子"
    assert len(set(all_seeds(BLOCK_A))) == 100 and len(set(all_seeds(BLOCK_B))) == 100
    assert len(set(all_seeds(BLOCK_PUBLIC))) == 4
    # 旧调用兼容
    assert list_seeds(GROUP_UNIFORM, 3) == (7001, 7002, 7003)
    assert list_seeds(GROUP_CROSSING) == CROSSING_SEEDS
    assert list_seeds() == UNIFORM_SEEDS + CROSSING_SEEDS
    assert groups() == {GROUP_UNIFORM: UNIFORM_SEEDS, GROUP_CROSSING: CROSSING_SEEDS}
    assert groups() is not _GROUPS
    assert describe() == describe_block(BLOCK_A).replace("块 A: ", "private: ") + f"; public={list(PUBLIC_SEEDS)}"
    # 新调用
    assert list_seeds(BLOCK_B, GROUP_UNIFORM, 3) == (5001, 5002, 5003)
    assert list_seeds(BLOCK_B) == UNIFORM_SEEDS_B + CROSSING_SEEDS_B
    assert list_seeds(BLOCK_PUBLIC) == PUBLIC_SEEDS
    assert group_of_public_seed(2001) == GROUP_CROSSING
    assert blocks() == ("A", "B", "public")
    assert block_seeds(BLOCK_B, 2) == {GROUP_UNIFORM: (5001, 5002), GROUP_CROSSING: (6001, 6002)}


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 输出流可能不可重配置
            pass
    _self_check()
    print(describe())
    print(describe_block(BLOCK_A))
    print(describe_block(BLOCK_B))
    print(describe_block(BLOCK_PUBLIC))
    print("self-check: ok")
