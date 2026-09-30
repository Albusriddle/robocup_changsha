# tools_local —— P666 本地实验基建

本目录只服务于**本地开发比较**：用官方环境和官方指标回答"版本 A 是否好于版本 B"。
它不参与正式评测，也不改变官方评测行为（`coverage_bench/`、`configs/` 一律只读）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `seedset.py` | 固定的私有开发种子集常量（先固定、再使用、永不用于训练）：uniform 7001..7050、crossing 8001..8050、公开 4 例 1001/1002/2001/2002。纯常量 + 纯函数，无随机抽样。 |
| `eval_mc.py` | 多种子蒙特卡洛评测：官方环境直跑，双口径（私有种子 / 公开 4 例），输出 J̄、S、CI95、每步覆盖率曲线、碰撞率、act 耗时 p95，写 JSON + CSV。 |
| `robustness_check.py` | 极端输入与跨局状态自测：全不可见、目标/队友精确重合、NaN/inf、`step_index` 跳变与复位、单实例连续 8 局复用；每条打印 PASS/FAIL。 |

## 运行（仓库根目录，Windows PowerShell）

```powershell
# 基线：私有 50 例/组 + 公开 4 例（8 局）
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\baseline --tag baseline --n-per-group 50

# 消融：覆盖 entry.DEFAULT_CONFIG 的键（走 entry.CoverageRulePolicy）
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\e001 --tag e001-unit --set unit_magnitude=true

# 只跑私有口径 / 自定义种子
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\probe --tag probe --no-public `
    --seeds-uniform 1001,1002,5001,5002 --seeds-crossing 2001,2002,6001,6002

# 鲁棒性自测
.\.venv\Scripts\python.exe participant\P666\tools_local\robustness_check.py
```

`--set` 可用键（与 `entry.DEFAULT_CONFIG` 逐字一致）：
`control_gain`、`predict_horizon`(`remaining`/`zero`)、`unit_magnitude`(bool)、`deconflict`(bool)、
`safety_radius`、`safety_gain`、`memory`(bool)、`blind_action`(`center`/`stay`)、`clip_eps`。

## 口径（与官方同构，不自造评分公式）

- 奖励与指标一律取 `infos[agent]["reward_terms"]` / `infos[agent]["metrics"]`（官方 metrics/rewards），
  本工具不重算 `C_t` / `K_t`。
- 每组先按 scenario 求均值、再对场景取平均；`S = 500 × (J̄_basic + J̄_coop)`。
- CI95 = bootstrap 1000 次、`seed=81273`（`SeedSequence(seed, spawn_key=(group_index,))`）、
  重采样 scenario 均值、取 2.5/97.5 分位；总分 CI 由各组 bootstrap 索引联合给出。
- 退出码：全部局 `status == "ok"` → 0；出现任何非 ok 局 → 3，并在 stdout 逐条打印失败局面。

## 已验证事实（2026-09-30，Python 3.12.10 / numpy 2.5.3 / mpe2 1.1.1）

1. **与官方评测器一致**：公开套件上本工具 `S = 221.667`，
   `scripts/evaluate_one.py` 的 `performance_score = 221.66666666666669`（`status=ok`）；
   逐组 `mean_j` 与 `ci95_j` 完全相同（basic 0.11667 / [0.0667, 0.1667]，cooperation 0.32667 / [0.0867, 0.5667]）。
2. **确定性**：同一命令重跑两次，108 局逐局 `return_sum` 最大差 `0.0`，S 完全相同。
3. **种子块依赖**：同一策略在 PLAN 探测用的种子块（`1001,1002 + 5001..5050` / `2001,2002 + 6001..6050`，52 例/组）
   上 `S = 133.654`，与 PLAN §2.5 的 133.0 吻合；在本目录固定的私有块（7001..7050 / 8001..8050，50 例/组）
   上 `S = 184.667`（当前冻结版 `entry.py`；独立复核已多次复现）。→ 绝对值只在**同一固定种子集内**可比，
   跨种子块的绝对值不可比。
   - 勘误（2026-09-30，独立复核发现）：本文早期把这一个数写成 `184.533`。`184.533` 实为
     `--set safety_radius=0.0` 那一档的块 A 值（与早期一次未提交修订的留档逐场景相同），**不是默认配置的基线**；
     默认配置的正确值是 `184.667`。此前 `entry.py` 曾被改动过若干轮，早期留档无法从 git 还原。
4. **鲁棒性**：`robustness_check.py` 12/12 PASS（含 NaN/inf 与 8 局复用 `max|Δa| = 0.0`）。
5. **静态审计**：`scripts/check_submission.py` → `rejections: []`，本目录三个脚本没有 `scan_hits`。
