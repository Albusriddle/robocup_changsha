r"""F2 · 对官方核验成绩的预测区间（套件级重抽样）。

为什么需要这个工具
------------------
`eval_mc.py` 报的是"50 例/组的 CI95"（块 B 半宽 ≈ ±25.6）。但组织方核验用的是**同构小套件**
（公开套件是 2 场景 × 2 重复/组），同一策略在公开套件上的 CI95 半宽达 ±145 —— 两者相差 5.7 倍。
也就是说，"50 例/组的 CI95"**不能**用来预测核验成绩。本工具用**实测重抽样**回答：

    "若核验套件是同构的（每组 2 个场景 × 2 次重复），同一策略的 S 会落在哪里？"

做法
----
1) 场景池：用 4 个固定、互不重叠的种子块（A/B/C/D，各 50 例 uniform + 50 例 crossing，
   共 200 + 200 个场景）跑一遍，得到每个场景的 J（逐场景只跑 1 次，确定性策略下重复无影响）。
2) 套件重抽样：每次从 uniform 池抽 2 个场景、从 crossing 池抽 2 个场景，各按公开口径重复 2 次，
   计 `S = 500·(J̄_basic + J̄_coop)`；重复 `--resamples` 次（默认 10000，固定种子）。
3) 输出 `P5 / P50 / P95` 与 `P(S ≥ 130)`。
4) 两层 bootstrap（先抽块、再抽块内场景）给出含块效应的保守区间，与"只抽场景"并列。

必须明写的假设
--------------
"私有核验套件与公开套件同构（每组 2 场景 × 2 重复）"是一个**假设**：组织方未公布私有套件的
规模与种子（开发笔记 10.4-2）。若后续公布，改 `--scenarios-per-group` / `--repeats` / `--groups` 重跑。

依赖
----
只依赖 numpy 与官方环境自带的 scipy/coverage_bench；**不使用**任何被静态审计禁止或需人工审核的符号（进程调用、动态导入、反序列化等）
禁用符号（提交目录内的 .py 会被静态审计）。复用 `eval_mc.py` 的 runner 与 `seedset.py` 的固定种子，
不另写评分公式。

用法（仓库根目录，评测环境）
--------------------------
    .\.venv\Scripts\python.exe participant\P666\tools_local\predict_score.py --output outputs\P666\predict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent          # participant/P666/tools_local
_SUBMISSION = _HERE.parent                       # participant/P666
_REPO = _HERE.parents[2]                         # 仓库根

for _p in (str(_HERE), str(_SUBMISSION), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_mc  # noqa: E402  # 复用它的 runner 与聚合口径（静态导入）
import seedset  # noqa: E402

SCORE_SCALE = 500.0            # S = 500 * (J_bar_basic + J_bar_coop)
DEFAULT_BLOCKS = ("A", "B", "C", "D")
DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260930
DEFAULT_SCENARIOS_PER_GROUP = 2
DEFAULT_REPEATS = 2
DEFAULT_GROUPS = "basic,cooperation"


def collect_pools(blocks: Sequence[str], suite_path: Path) -> Tuple[List[float], List[float], Dict[str, List[float]]]:
    """跑各块的场景，返回 (uniform 池 J, crossing 池 J, 分块池)。"""
    suite = eval_mc.load_suite(suite_path) if hasattr(eval_mc, "load_suite") else None
    if suite is None:  # 兜底：eval_mc 内部持有加载函数
        from coverage_bench.suites import load_suite as _ls
        suite = _ls(suite_path)
    factory, _prov = eval_mc.make_policy_factory({})
    uniform_pool: List[float] = []
    crossing_pool: List[float] = []
    per_block: Dict[str, List[float]] = {}
    for block in blocks:
        seeds_by_group = seedset.block_seeds(block)
        private_cases, _public = eval_mc.build_plan(suite, seeds_by_group, include_public=False)
        records = eval_mc.run_plan(private_cases, factory, block_label=block)
        bad = [r for r in records if r.get("status") != "ok"]
        if bad:
            raise SystemExit(f"块 {block} 有 {len(bad)} 局非 ok，拒绝据此给出预测区间")
        js: List[float] = []
        for rec in records:
            j = float(rec["mean_j"])
            js.append(j)
            if str(rec["group_id"]).startswith("basic"):
                uniform_pool.append(j)
            else:
                crossing_pool.append(j)
        per_block[block] = js
        print(f"  块 {block}: {len(records)} 局，J 均值 {np.mean(js):.4f}")
    return uniform_pool, crossing_pool, per_block


def sample_suite(rng: np.random.Generator, u_pool: Sequence[float], c_pool: Sequence[float],
                 n_per_group: int, repeats: int) -> float:
    """抽一个同构小套件并打分（确定性策略下 repeats 不改变 J，但保留口径）。"""
    u = rng.choice(len(u_pool), size=n_per_group, replace=True)
    c = rng.choice(len(c_pool), size=n_per_group, replace=True)
    jb = float(np.mean([u_pool[i] for i in u]))
    jc = float(np.mean([c_pool[i] for i in c]))
    _ = repeats  # 口径占位：重复次数对确定性策略不改变 J
    return SCORE_SCALE * (jb + jc)


def pct(values: np.ndarray) -> Dict[str, float]:
    return {"P5": float(np.percentile(values, 5)),
            "P50": float(np.percentile(values, 50)),
            "P95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P666 官方核验成绩预测区间（套件级重抽样）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（写 predict-<tag>.json）")
    parser.add_argument("--tag", type=str, default="predict")
    parser.add_argument("--blocks", type=str, default=",".join(DEFAULT_BLOCKS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenarios-per-group", type=int, default=DEFAULT_SCENARIOS_PER_GROUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--groups", type=str, default=DEFAULT_GROUPS)
    parser.add_argument("--threshold", type=float, default=130.0)
    parser.add_argument("--suite", type=str, default=str(_REPO / "configs" / "public-suite-v1.yaml"))
    args = parser.parse_args(argv)

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    for b in blocks:
        if b not in seedset.blocks():
            raise SystemExit(f"未知种子块 {b!r}，可用 {seedset.blocks()}")

    print("=" * 78)
    print("F2 预测区间：对官方核验成绩的套件级重抽样")
    print("=" * 78)
    print("假设：私有核验套件与公开套件同构")
    print(f"      --groups={args.groups}（本仓库冻结套件只有 basic/cooperation 两组；"
          f"若组织方公布更多组，改此参数重跑）")
    print(f"      每组 {args.scenarios_per_group} 个场景 × {args.repeats} 次重复；"
          f"重抽样 {args.resamples} 次，seed={args.seed}")
    print("说明：本策略确定性 ⇒ 重复次数不改变 J；保留 repeats 只为保持打分口径正确。")
    print("\n[1/3] 采集场景池（4 个固定块，互不重叠）")
    u_pool, c_pool, _per_block = collect_pools(blocks, Path(args.suite))
    print(f"  uniform 池={len(u_pool)} 场景，crossing 池={len(c_pool)} 场景")

    rng = np.random.default_rng(args.seed)

    print("\n[2/3] 只抽场景（不抽块）：等价于'核验套件从全部场景池里抽'")
    scene_scores = np.array([sample_suite(rng, u_pool, c_pool, args.scenarios_per_group, args.repeats)
                             for _ in range(args.resamples)])
    scene_stats = pct(scene_scores)
    p_ge = float(np.mean(scene_scores >= args.threshold))
    print(f"  P5={scene_stats['P5']:.1f}  P50={scene_stats['P50']:.1f}  P95={scene_stats['P95']:.1f}  "
          f"mean={scene_stats['mean']:.1f}  P(S>={args.threshold:g})={p_ge:.3f}")

    print("\n[3/3] 两层 bootstrap：先抽块、再抽块内场景（含块效应的保守区间）")
    per_block_u: Dict[str, List[float]] = {}
    per_block_c: Dict[str, List[float]] = {}
    idx_u = 0
    idx_c = 0
    for block in blocks:
        n_u = len(seedset.block_seeds(block)["uniform"])
        n_c = len(seedset.block_seeds(block)["crossing"])
        per_block_u[block] = u_pool[idx_u:idx_u + n_u]
        per_block_c[block] = c_pool[idx_c:idx_c + n_c]
        idx_u += n_u
        idx_c += n_c

    # 每次重抽样只取一个块：模拟"核验套件的种子整体落在某一类几何区域"的情形，
    # 因此区间同时包含块内噪声与块间差异（比只抽场景更保守）。
    two_level = np.empty(args.resamples, dtype=float)
    for r in range(args.resamples):
        blk = blocks[int(rng.integers(0, len(blocks)))]
        ub = per_block_u[blk]
        cb = per_block_c[blk]
        us = [ub[int(rng.integers(0, len(ub)))] for _ in range(args.scenarios_per_group)]
        cs = [cb[int(rng.integers(0, len(cb)))] for _ in range(args.scenarios_per_group)]
        two_level[r] = SCORE_SCALE * (float(np.mean(us)) + float(np.mean(cs)))
    tl_stats = pct(two_level)
    tl_p_ge = float(np.mean(two_level >= args.threshold))
    print(f"  P5={tl_stats['P5']:.1f}  P50={tl_stats['P50']:.1f}  P95={tl_stats['P95']:.1f}  "
          f"mean={tl_stats['mean']:.1f}  P(S>={args.threshold:g})={tl_p_ge:.3f}")

    payload = {
        "tool": "predict_score.py",
        "purpose": "官方核验成绩的预测区间（套件级重抽样）",
        "assumption": {
            "suite_isomorphic_to_public": True,
            "scenarios_per_group": args.scenarios_per_group,
            "repeats": args.repeats,
            "groups": args.groups,
            "note": "私有核验套件的规模与种子未公布；本假设需在公布后复核并重跑。",
        },
        "deterministic_policy_note": "策略确定性 ⇒ repeats 不改变 J，保留仅为口径正确。",
        "blocks": blocks,
        "scenario_pool": {"uniform": len(u_pool), "crossing": len(c_pool)},
        "resamples": args.resamples,
        "seed": args.seed,
        "threshold": args.threshold,
        "scene_only": {**scene_stats, "P_ge_threshold": p_ge},
        "two_level_bootstrap": {**tl_stats, "P_ge_threshold": tl_p_ge},
        "block_inner_ci95_for_reference": {
            "note": "eval_mc 的 50 例/组 CI95 仅衡量块内均值不确定性，不能预测核验成绩。",
        },
    }
    print("\n=== 结论 ===")
    print(f"  块内 50 例/组 CI95（旧口径，仅作对照，不能预测核验成绩）")
    print(f"  套件级预测区间（只抽场景）：P5={scene_stats['P5']:.1f} / P50={scene_stats['P50']:.1f} / "
          f"P95={scene_stats['P95']:.1f}，P(S>={args.threshold:g})={p_ge:.3f}")
    print(f"  套件级预测区间（两层 bootstrap，含块效应）：P5={tl_stats['P5']:.1f} / P50={tl_stats['P50']:.1f} / "
          f"P95={tl_stats['P95']:.1f}，P(S>={args.threshold:g})={tl_p_ge:.3f}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"predict-{args.tag}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
