r"""S3 · 对官方核验成绩的预测区间（多块场景池 + 三种重抽样模型）。

为什么需要这个工具
------------------
`eval_mc.py` 报的是"50 例/组的 CI95"（块 B 半宽 ≈ ±25.6）。但组织方核验用的是**同构小套件**
（公开套件是 2 场景 × 2 重复/组），同一策略在公开套件上的 CI95 半宽达 ±145 —— 相差 5 倍以上。
也就是说，"50 例/组的 CI95"**不能**用来预测核验成绩。本工具用**实测重抽样**回答：

    "若核验套件是同构的（每组 2 个场景 × 2 次重复），同一策略的 S 会落在哪里？"

三种模型（并列输出，**不预设谁更保守**，以实测宽度为准）
----------------------------------------------------
1. `scene_only`（场景重抽样）：把 N 个块的场景汇成一个池，每次抽 2 个 basic 场景 + 2 个
   cooperation 场景，重复 `--repeats` 次后计 `S = 500·(J̄_basic + J̄_coop)`。
2. `block_random_effects`（块间随机效应模型）：每次**从 N 个块中均匀取 1 块**，再在该块内抽
   2 + 2 个场景。它把"核验套件的种子整体落在哪一类几何区域"这一层不确定性也算了进去。
3. `two_level_standard`（标准两层 bootstrap）：第 1 层把 N 个块**有放回**重抽，第 2 层在抽中的
   块内抽场景；等价于从"块多重度加权后的场景池"里抽 2 + 2 个场景。

另附：块均值表、方差分解（块内 vs 块间）、留一块交叉验证（逐个剔除一个块后重算区间），
用来量化"区间对块选择的敏感度"。S1 的 `sys.dont_write_bytecode` 自守与 S4 的工具/提交归因也在本文件。

必须明写的假设
--------------
"私有核验套件与公开套件同构（每组 2 场景 × 2 重复）"是一个**假设**：组织方未公布私有套件的
规模与种子。若后续公布，改 `--scenarios-per-group` / `--repeats` / `--groups` 重跑。

用法（仓库根目录，评测环境）
--------------------------
    .\.venv\Scripts\python.exe participant\P666\tools_local\predict_score.py --output outputs\P666\predict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True  # PYC_GUARD：不写出 __pycache__（官方预检把被忽略文件也算未提交内容）

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
DEFAULT_BLOCKS = tuple(b for b in seedset.blocks() if b != seedset.BLOCK_PUBLIC)
DEFAULT_RESAMPLES = 10000
DEFAULT_LOO_RESAMPLES = 2000
DEFAULT_SEED = 20260930
DEFAULT_SCENARIOS_PER_GROUP = 2
DEFAULT_REPEATS = 2
DEFAULT_GROUPS = "basic,cooperation"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_pools(blocks: Sequence[str], suite_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """跑各块场景，返回 (uniform 矩阵 (B,50), crossing 矩阵 (B,50), 逐块摘要)。"""
    from coverage_bench.suites import load_suite as _ls
    suite = _ls(suite_path)
    factory, _prov = eval_mc.make_policy_factory({})
    u_rows: List[List[float]] = []
    c_rows: List[List[float]] = []
    per_block: Dict[str, Dict[str, float]] = {}
    for block in blocks:
        seeds_by_group = seedset.block_seeds(block)
        private_cases, _public = eval_mc.build_plan(suite, seeds_by_group, include_public=False)
        records = eval_mc.run_plan(private_cases, factory, block_label=block)
        bad = [r for r in records if r.get("status") != "ok"]
        if bad:
            raise SystemExit(f"块 {block} 有 {len(bad)} 局非 ok，拒绝据此给出预测区间")
        u = [float(r["mean_j"]) for r in records if str(r["group_id"]).startswith("basic")]
        c = [float(r["mean_j"]) for r in records if not str(r["group_id"]).startswith("basic")]
        u_rows.append(u)
        c_rows.append(c)
        allj = u + c
        per_block[block] = {
            "uniform_mean_j": float(np.mean(u)),
            "crossing_mean_j": float(np.mean(c)),
            "mean_j": float(np.mean(allj)),
            "S_50x50": SCORE_SCALE * (float(np.mean(u)) + float(np.mean(c))),
            "n": len(allj),
        }
        print(f"  块 {block}: {len(records)} 局  J_basic={np.mean(u):.4f}  J_coop={np.mean(c):.4f}  "
              f"S(50/50)={per_block[block]['S_50x50']:.1f}")
    return np.asarray(u_rows, dtype=float), np.asarray(c_rows, dtype=float), per_block


def variance_decomposition(u: np.ndarray, c: np.ndarray) -> Dict[str, Dict[str, float]]:
    """场景级 J 的方差分解：块间 vs 块内（ANOVA 平方和口径）。"""
    out: Dict[str, Dict[str, float]] = {}
    for name, mat in (("uniform", u), ("crossing", c), ("pooled", np.concatenate([u, c], axis=1))):
        block_means = mat.mean(axis=1)
        grand = mat.mean()
        n_per = mat.shape[1]
        ss_between = float(n_per * np.sum((block_means - grand) ** 2))
        ss_within = float(np.sum((mat - block_means[:, None]) ** 2))
        total = ss_between + ss_within
        out[name] = {
            "grand_mean_j": float(grand),
            "block_mean_min": float(block_means.min()),
            "block_mean_max": float(block_means.max()),
            "block_mean_std": float(block_means.std(ddof=1)),
            "ss_between": ss_between,
            "ss_within": ss_within,
            "between_share": (ss_between / total) if total > 0 else float("nan"),
            "within_share": (ss_within / total) if total > 0 else float("nan"),
        }
    return out


def sample_models(rng: np.random.Generator, u: np.ndarray, c: np.ndarray,
                  resamples: int, n_per_group: int) -> Dict[str, np.ndarray]:
    """返回三种模型的 S 分布（每模型 resamples 个数）。"""
    nb, ncol = u.shape
    uf, cf = u.reshape(-1), c.reshape(-1)          # 池化（顺序 = 块顺序）
    n_pool = uf.size

    # 1) 只抽场景
    iu = rng.integers(0, n_pool, size=(resamples, n_per_group))
    ic = rng.integers(0, n_pool, size=(resamples, n_per_group))
    scene_only = SCORE_SCALE * (uf[iu].mean(axis=1) + cf[ic].mean(axis=1))

    # 2) 块间随机效应：每次均匀取 1 块
    b1 = rng.integers(0, nb, size=resamples)
    col1 = rng.integers(0, ncol, size=(resamples, n_per_group))
    bu = u[b1]
    bc = c[b1]
    bre = SCORE_SCALE * (np.take_along_axis(bu, col1, axis=1).mean(axis=1)
                         + np.take_along_axis(bc, col1, axis=1).mean(axis=1))

    # 3) 标准两层 bootstrap：块有放回重抽（第 1 层），再在抽中的块内抽场景（第 2 层）
    bb = rng.integers(0, nb, size=(resamples, nb))
    slot = rng.integers(0, nb, size=(resamples, n_per_group))
    bidx = np.take_along_axis(bb, slot, axis=1)
    col2 = rng.integers(0, ncol, size=(resamples, n_per_group))
    tl_u = u[bidx, col2]
    tl_c = c[bidx, col2]
    two_level = SCORE_SCALE * (tl_u.mean(axis=1) + tl_c.mean(axis=1))

    return {"scene_only": scene_only, "block_random_effects": bre, "two_level_standard": two_level}


def summarize(values: np.ndarray, threshold: float) -> Dict[str, float]:
    return {
        "P5": float(np.percentile(values, 5)),
        "P25": float(np.percentile(values, 25)),
        "P50": float(np.percentile(values, 50)),
        "P75": float(np.percentile(values, 75)),
        "P95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "P_ge_threshold": float(np.mean(values >= threshold)),
    }


def leave_one_out(rng: np.random.Generator, u: np.ndarray, c: np.ndarray, blocks: Sequence[str],
                  resamples: int, n_per_group: int, threshold: float) -> Dict[str, Dict[str, float]]:
    """逐个剔除一个块后重算块间随机效应模型的分位数（区间对块选择的敏感度）。"""
    out: Dict[str, Dict[str, float]] = {}
    nb = u.shape[0]
    for i in range(nb):
        keep = [j for j in range(nb) if j != i]
        dist = sample_models(rng, u[keep], c[keep], resamples, n_per_group)
        vals = dist["block_random_effects"]
        out[blocks[i]] = {
            "P5": float(np.percentile(vals, 5)),
            "P50": float(np.percentile(vals, 50)),
            "P95": float(np.percentile(vals, 95)),
            "P_ge_threshold": float(np.mean(vals >= threshold)),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P666 官方核验成绩预测区间（多块场景池 + 三种重抽样模型）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（写 predict-<tag>.json）")
    parser.add_argument("--tag", type=str, default="predict")
    parser.add_argument("--blocks", type=str, default=",".join(DEFAULT_BLOCKS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--loo-resamples", type=int, default=DEFAULT_LOO_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenarios-per-group", type=int, default=DEFAULT_SCENARIOS_PER_GROUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--groups", type=str, default=DEFAULT_GROUPS)
    parser.add_argument("--threshold", type=float, default=130.0)
    parser.add_argument("--suite", type=str, default=str(_REPO / "configs" / "public-suite-v1.yaml"))
    parser.add_argument("--code-commit", type=str, default=None,
                        help="S4：代码提交号（由调用方执行 git rev-parse HEAD 后传入；默认 null=未提供）")
    parser.add_argument("--no-loo", action="store_true", help="跳过留一块交叉验证（更快）")
    args = parser.parse_args(argv)

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    for b in blocks:
        if b not in seedset.blocks() or b == seedset.BLOCK_PUBLIC:
            raise SystemExit(f"未知/不可用种子块 {b!r}，可用: {[x for x in seedset.blocks() if x != 'public']}")
    if args.resamples <= 0:
        raise SystemExit("--resamples 必须为正整数")
    if args.scenarios_per_group <= 0:
        raise SystemExit("--scenarios-per-group 必须为正整数")

    # 输出目录探活：独立复核条目 7-③ 发现原先会先跑完 16 个块、再在写盘阶段抛
    # 未捕获的 FileNotFoundError。这里提前干净报错（exit 2）。
    out_dir: Path | None = None
    if args.output:
        out_dir = Path(args.output)
        if not out_dir.is_absolute():
            out_dir = _REPO / out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe = out_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            print(f"!! 输出目录不可写: {out_dir}  ({type(exc).__name__}: {exc})")
            return 2

    print("=" * 84)
    print("S3 预测区间：多块场景池 + 三种重抽样模型")
    print("=" * 84)
    print("假设：私有核验套件与公开套件同构")
    print(f"      --groups={args.groups}（本仓库冻结套件只有 basic/cooperation 两组；"
          f"若组织方公布更多组，改此参数重跑）")
    print(f"      每组 {args.scenarios_per_group} 个场景 x {args.repeats} 次重复；"
          f"块数 {len(blocks)}；重抽样 {args.resamples} 次，seed={args.seed}")
    print("说明：本策略确定性 => 重复次数不改变 J；保留 repeats 只为保持打分口径正确。")
    print("      '更保守'不作预设：三个模型的宽度由下面实测结果说话。")

    print(f"\n[1/4] 采集场景池（{len(blocks)} 个固定块，互不重叠）")
    u, c, per_block = collect_pools(blocks, Path(args.suite))
    print(f"  uniform 池={u.size} 场景，crossing 池={c.size} 场景")

    print("\n[2/4] 块均值表（每块 50+50 场景；绝对值不可跨块比较）")
    print(f"  {'块':<3} {'J_basic':>9} {'J_coop':>9} {'J_all':>8} {'S(50/50)':>9}")
    for b in blocks:
        d = per_block[b]
        print(f"  {b:<3} {d['uniform_mean_j']:>9.4f} {d['crossing_mean_j']:>9.4f} "
              f"{d['mean_j']:>8.4f} {d['S_50x50']:>9.1f}")

    vd = variance_decomposition(u, c)
    print("\n  方差分解（场景级 J 的 ANOVA 平方和）")
    for name in ("uniform", "crossing", "pooled"):
        d = vd[name]
        print(f"    {name:<9} 块均值∈[{d['block_mean_min']:.4f},{d['block_mean_max']:.4f}] "
              f"块间占比={d['between_share']*100:.1f}%  块内占比={d['within_share']*100:.1f}%")

    rng = np.random.default_rng(args.seed)
    print("\n[3/4] 三种重抽样模型（同一随机种子、同一重抽样次数）")
    dist = sample_models(rng, u, c, args.resamples, args.scenarios_per_group)
    stats = {k: summarize(v, args.threshold) for k, v in dist.items()}
    for k in ("scene_only", "block_random_effects", "two_level_standard"):
        s = stats[k]
        print(f"  {k:<22} P5={s['P5']:7.1f}  P50={s['P50']:7.1f}  P95={s['P95']:7.1f}  "
              f"sd={s['sd']:6.1f}  P(S>={args.threshold:g})={s['P_ge_threshold']:.3f}")
    width = {k: stats[k]["P95"] - stats[k]["P5"] for k in stats}
    widest = max(width, key=lambda k: width[k])
    print(f"  => 实测 P5-P95 跨度: { {k: round(v, 1) for k, v in width.items()} }; "
          f"最宽的是 {widest}（描述以实测宽度为准，不预设谁更保守）")

    loo: Dict[str, Dict[str, float]] = {}
    if not args.no_loo:
        print(f"\n[4/4] 留一块交叉验证（逐个剔除 1 块后重算块间随机效应模型，各 {args.loo_resamples} 次）")
        loo = leave_one_out(np.random.default_rng(args.seed + 1), u, c, blocks,
                            args.loo_resamples, args.scenarios_per_group, args.threshold)
        p50s = [v["P50"] for v in loo.values()]
        p95s = [v["P95"] for v in loo.values()]
        pges = [v["P_ge_threshold"] for v in loo.values()]
        print(f"  剔除任一单块后: P50∈[{min(p50s):.1f},{max(p50s):.1f}]  "
              f"P95∈[{min(p95s):.1f},{max(p95s):.1f}]  P(S>={args.threshold:g})∈[{min(pges):.3f},{max(pges):.3f}]")
        base_p50 = stats["block_random_effects"]["P50"]
        worst = max(loo, key=lambda b: abs(loo[b]["P50"] - base_p50))
        print(f"  对 P50 影响最大的块: {worst}（ΔP50={loo[worst]['P50'] - base_p50:+.1f}）")

    tool_path = Path(__file__).resolve()
    entry_path = _SUBMISSION / "entry.py"
    payload = {
        "tool": "predict_score.py",
        "purpose": "官方核验成绩的预测区间（多块场景池 + 三种重抽样模型）",
        "assumption": {
            "suite_isomorphic_to_public": True,
            "scenarios_per_group": args.scenarios_per_group,
            "repeats": args.repeats,
            "groups": args.groups,
            "note": "私有核验套件的规模与种子未公布；本假设需在公布后复核并重跑。",
        },
        "deterministic_policy_note": "策略确定性 => repeats 不改变 J，保留仅为口径正确。",
        "blocks": blocks,
        "n_blocks": len(blocks),
        "scenario_pool": {"uniform": int(u.size), "crossing": int(c.size)},
        "resamples": args.resamples,
        "seed": args.seed,
        "threshold": args.threshold,
        "block_means": per_block,
        "variance_decomposition": vd,
        # ---- 补丁 A（R3）：落盘逐场景 J，使预测区间可被第三方独立复算 ----
        # 说明：产物此前只有块均值与汇总统计，"从产物独立复算 P5/P50/P95" 物理上不可执行。
        # 现在给出 16 块 × 50 场景 × 2 组的原始 J 与对应种子，任何第三方都能用自写 bootstrap 复算。
        "per_scenario_j": {
            "layout": "uniform/crossing 均为 (块数, 50)；行序 = blocks；列序 = 该块的场景顺序（种子见 *_seeds）",
            "blocks": blocks,
            "uniform_seeds": {b: list(seedset.block_seeds(b)["uniform"]) for b in blocks},
            "crossing_seeds": {b: list(seedset.block_seeds(b)["crossing"]) for b in blocks},
            "uniform": [[float(v) for v in row] for row in u],
            "crossing": [[float(v) for v in row] for row in c],
        },
        "sampling_scheme": {
            "scores": "S = 500 * (mean(J_basic) + mean(J_coop))",
            "scene_only": "每组从该组池（B*50 个场景，行优先）等概率有放回抽 scenarios_per_group 个",
            "block_random_effects": "每次先均匀抽 1 个块（0..B-1），再在该块的组内等概率有放回抽 scenarios_per_group 个",
            "two_level_standard": "第 1 层：从 B 个块中有放回抽 B 个（得长度 B 的块索引序列）；"
                                  "第 2 层：每个 replicate 从这 B 个槽位里等概率抽 scenarios_per_group 个槽位，"
                                  "取该槽位的块，再在该块内抽场景",
            "rng": "numpy.random.default_rng(seed)；三个模型依次使用同一 rng 流（顺序：scene_only → block_random_effects → two_level_standard）",
            "note": "复算时须按上述顺序与参数使用同一 rng，才能逐位复现；不同实现只要抽样方案一致，分位数应一致。",
        },
        "models": stats,
        "model_widths_P95_minus_P5": width,
        "widest_model": widest,
        "leave_one_out": loo,
        "provenance": {
            "tool_path": str(tool_path),
            "tool_sha256": _sha256(tool_path),
            "seedset_path": str(Path(seedset.__file__).resolve()),
            "seedset_sha256": _sha256(Path(seedset.__file__).resolve()),
            "entry_path": str(entry_path),
            "entry_sha256": _sha256(entry_path) if entry_path.is_file() else None,
            "code_commit": args.code_commit if args.code_commit else None,
            "code_commit_note": ("由调用方注入（acceptance.ps1 执行 git rev-parse HEAD）"
                                 if args.code_commit else "未提供：请用 --code-commit 传入 git rev-parse HEAD"),
        },
    }
    if out_dir is not None:
        path = out_dir / f"predict-{args.tag}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
