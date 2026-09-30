# tools_local —— P666 本地实验基建

本目录只服务于**本地开发比较**：用官方环境和官方指标回答"版本 A 是否好于版本 B"。
它不参与正式评测，也不改变官方评测行为（`coverage_bench/`、`configs/` 一律只读）。
**本地工具依赖**（numpy / scipy 由官方环境提供）**不是推理依赖**，因此不写入 `requirements-infer.lock`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `seedset.py` | 固定的私有开发种子集常量（先固定、再使用、永不用于训练）。四块：A `7001..7050 / 8001..8050`、B `5001..5050 / 6001..6050`、C `9001..9050 / 9101..9150`、D `9201..9250 / 9301..9350`，另含公开 4 例。纯常量 + 纯函数，无随机抽样，自带 self-check。 |
| `eval_mc.py` | 多种子蒙特卡洛评测：官方环境直跑，双口径（私有种子 / 公开 4 例），输出 J̄、S、CI95、每步覆盖率曲线、碰撞率、act 耗时 p95，写 JSON + CSV；含 **F11 策略内部异常计数断言**。 |
| `predict_score.py` | **F2**：对**官方核验成绩**的预测区间。四块共 200+200 场景池 → 套件级重抽样（默认 10000 次）→ `P5/P50/P95` 与 `P(S≥130)`，并给出含块效应的两层 bootstrap 区间。 |
| `robustness_check.py` | 极端输入与跨局状态自测（15 条）：全不可见、目标/队友精确重合、NaN/inf、`step_index` 跳变与复位、单实例连续 8 局复用、close、以及 **F11 静默降级可观测性**。 |
| `acceptance.ps1` | **F15** 一键验收：预检 → 公开评测 → 双块 MC → 鲁棒性 → 范围自查，逐项打印退出码，任一失败返回非零。 |
| `real_dynamics_probe.py` | **F8** 证据脚本：实测机器人位移曲线、目标运动、奖励结构、接触力（本机复现 10 步轴向 0.2490 m）。 |
| `drive_magnitude_probe.py` | **F8** 证据脚本：动作幅度 → 位移（对角满推 `|u|=√2` 得 0.3522 m）与三种控制律对照。 |
| `geometric_ceiling.py` | **F8** 证据脚本：与策略无关的几何上界（真值目标轨迹 + 最优固定朝向 + 逐步最大匹配）。 |

## 运行（仓库根目录，Windows PowerShell）

```powershell
# 双块 MC（块 A 与 块 B + 公开 4 例）
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\baseline --tag baseline --seed-block both

# 消融：覆盖 entry.DEFAULT_CONFIG 的键（走 entry.CoverageRulePolicy）
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\e007 --tag e007 --seed-block both --set predict_horizon=zero

# F2：官方核验成绩预测区间
.\.venv\Scripts\python.exe participant\P666\tools_local\predict_score.py --output outputs\P666\predict

# 鲁棒性自测（15 条，含 F11 断言）
.\.venv\Scripts\python.exe participant\P666\tools_local\robustness_check.py

# F11 自检：注入一次内部异常，证明断言会触发（期望退出码 3）
.\.venv\Scripts\python.exe participant\P666\tools_local\eval_mc.py `
    --output outputs\P666\inject --tag inject --n-per-group 2 --inject-error-probe 1 --no-public

# 一键验收
powershell -ExecutionPolicy Bypass -File participant\P666\tools_local\acceptance.ps1
```

`--set` 可用键（与 `entry.DEFAULT_CONFIG` 逐字一致）：
`control_gain`、`predict_horizon`(`remaining`/`zero`)、`unit_magnitude`(bool)、`deconflict`(bool)、
`safety_radius`、`safety_gain`、`memory`(bool)、`blind_action`(`center`/`stay`)、`clip_eps`。

## 口径（与官方同构，不自造评分公式）

- 奖励与指标一律取 `infos[agent]["reward_terms"]` / `infos[agent]["metrics"]`（官方 metrics/rewards），
  本工具不重算 `C_t` / `K_t`。
- 每组先按 scenario 求均值、再对场景取平均；`S = 500 × (J̄_basic + J̄_coop)`。
- **块内 CI95**（`eval_mc.py`）= bootstrap 1000 次、`seed=81273`、重采样 scenario 均值、取 2.5/97.5 分位；
  它只衡量**块内均值**的不确定性。
- **套件级预测区间**（`predict_score.py`）= 模拟"核验套件只含 2 场景/组"的口径，
  与块内 CI95 是**两个不同口径**：前者宽得多（本任务下 P5–P95 跨度 ≈ 260 分），
  后者约 ±25 分。**不能用块内 CI95 预测核验成绩。**
- 退出码：全部局 `status == "ok"` 且策略内部异常计数为 0 → 0；否则 3，并在 stdout 逐条打印失败局面。

## 已验证事实（2026-09-30，Python 3.12.10 / numpy 2.5.3 / mpe2 1.1.1）

1. **与官方评测器一致**：公开套件上本工具 `S = 221.667`，
   `scripts/evaluate_one.py` 的 `performance_score = 221.66666666666669`（`status=ok`）；
   逐组 `mean_j` 与 `ci95_j` 完全相同（basic 0.11667 / [0.0667, 0.1667]，cooperation 0.32667 / [0.0867, 0.5667]）。
2. **确定性**：同一命令重跑，208 局逐局 `return_sum` 最大差 `0.0`，S 完全相同。
3. **种子块依赖**：同一策略在 PLAN 探测块（`1001,1002 + 5001..5050` / `2001,2002 + 6001..6050`，52 例/组）
   上 `S = 133.654`；在本目录固定块上：**块 A `184.667`**、**块 B `130.267`**（当前冻结版 `entry.py`，独立复核已多次复现）。
   → 绝对值只在**同一固定种子集内**可比，跨种子块的绝对值不可比。
   - 勘误（2026-09-30，独立复核发现）：本文早期把块 A 基线写成 `184.533`。`184.533` 实为
     `--set safety_radius=0.0` 那一档的值，**不是默认配置基线**；默认配置的正确值是 `184.667`。
4. **鲁棒性**：`robustness_check.py` **15/15 PASS**（原 12 条 + F11 新增 3 条：计数器可读、
   畸形观测递增、reset 后归零）。
5. **静态审计**：`scripts/check_submission.py` → `rejections: []`、`scan_hits: []`（提交目录 19 个文件）。
6. **F2 预测区间**（四块 A/B/C/D，200+200 场景池）：只抽场景 `P5=41.7 / P50=158.3 / P95=305.1`，
   `P(S≥130)=0.649`；两层 bootstrap（含块效应）`P5=41.7 / P50=158.3 / P95=301.7`，`P(S≥130)=0.644`。
   假设：私有核验套件与公开套件同构（2 场景 × 2 重复/组）——组织方未公布私有规模，公布后需重跑。
7. **F8 移植脚本可复现关键数字**：`real_dynamics_probe.py` 实测 10 步轴向位移 0.2490 m；
   `drive_magnitude_probe.measure_displacement([1,1])` 得 0.3522 m；`geometric_ceiling.py` 得 ≈142.1。
8. **F3 诊断计数器**（`entry.stats`）：6000 次 act 中 `memory_hits=3`（块 A 0、块 B 3），
   `error_count=0`；3 次触发全在 uniform seed 5049 agent 1 step 1–3，且两条路径动作逐位相同 `[1.0, -1.0]`。
9. **F5/F6 机制统计**见 `LOG.md`：6240 次 act 中 66.603% 双分量饱和、96.074% 至少一分量饱和；
   盲走步占 41.35%、全盲 agent-局占 32.83%、t=0 平均可见目标 0.6567。
