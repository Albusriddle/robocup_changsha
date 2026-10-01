# tools_local —— P666 本地实验基建

本目录只服务于**本地开发比较**：用官方环境和官方指标回答"版本 A 是否好于版本 B"。
它不参与正式评测，也不改变官方评测行为（`coverage_bench/`、`configs/` 一律只读）。
**本地工具依赖**（numpy / scipy 由官方环境提供）**不是推理依赖**，因此不写入 `requirements-infer.lock`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `seedset.py` | 固定的私有开发种子集常量（先固定、再使用、永不用于训练）。**16 个私有块**：A `7001..7050 / 8001..8050`、B `5001..5050 / 6001..6050`、C `9001..9050 / 9101..9150`、D `9201..9250 / 9301..9350`、E–P `10001..12350`（每块 50+50），另含公开 4 例。纯常量 + 纯函数，无随机抽样，自带 self-check（含"块间与公开种子零重叠"断言）。 |
| `eval_mc.py` | 多种子蒙特卡洛评测：官方环境直跑，双口径（私有种子 / 公开 4 例），输出 J̄、S、CI95、每步覆盖率曲线、碰撞率、act 耗时 p95，写 JSON + CSV；含 **F11 策略内部异常计数断言**与 **S4 工具/提交归因字段**。 |
| `predict_score.py` | **S3**：对**官方核验成绩**的预测区间。**16 个固定块**（800+800 场景池）→ **三种重抽样模型并列**（`scene_only` / `block_random_effects` / `two_level_standard`）→ `P5/P50/P95` 与 `P(S≥130)`；另给块均值表、方差分解（块内 vs 块间）与留一块交叉验证。 |
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

# S3：官方核验成绩预测区间（16 块 + 三种重抽样模型 + 块均值/方差分解/留一法）
#     --code-commit 由调用方注入（acceptance.ps1 就是这么做的）
.\.venv\Scripts\python.exe participant\P666\tools_local\predict_score.py `
    --output outputs\P666\predict --code-commit (git rev-parse HEAD)

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
  与块内 CI95 是**两个不同口径**：前者宽得多（本任务下 P5–P95 跨度 ≈ 258 分），
  后者约 ±25 分。**不能用块内 CI95 预测核验成绩。**
- 三种模型**不预设谁更保守**（实测 P5–P95 跨度 258.3 / 255.0 / 258.3，块效应模型反而略窄）；
  场景级 J 的块间方差占比只有 0.9%–3.3%，即在"每组 2 场景"的尺度上块效应被场景内噪声淹没。
- **洁净语义（S1）**：官方预检用 `git status --porcelain --ignored -- .`（cwd 限定提交目录），
  **被忽略的文件同样算"未提交内容"**（`coverage_bench/submission.py:144-161`）。
  因此提交目录内**不得残留 `__pycache__/*.pyc`**：本目录每个脚本都在 `import sys` 后写了
  `sys.dont_write_bytecode = True`（不依赖环境变量），`acceptance.ps1` 也会检查该目录
  `*.pyc` 数为 0 且 `--ignored` 洁净检查为空，两项都计入退出码。
  通过时：预检 `working_tree_dirty: false`、`file_count` = `git ls-files participant/P666` 条数、
  `total_bytes` = 这些文件字节和。
- 退出码：全部局 `status == "ok"` 且策略内部异常计数为 0 → 0；否则 3，并在 stdout 逐条打印失败局面。

## 如何独立复算预测区间（补丁 A）

`predict-<tag>.json` 现在包含**逐场景 J**，因此第三方不需要重跑策略也能复算：

1. 读 `per_scenario_j`：`uniform` / `crossing` 均为 `(16, 50)` 的二维数组（行序 = `blocks`，
   列序 = 该块的场景顺序，种子见 `uniform_seeds` / `crossing_seeds`）。
2. 计分口径：`S = 500 × (mean(J_basic) + mean(J_coop))`。
3. 按 `sampling_scheme` 里写明的三种抽样方案各自实现 bootstrap（`--resamples`、`--seed` 也记录在产物里）：
   - `scene_only`：每组从该组池（`16×50` 个，行优先）等概率有放回抽 2 个；
   - `block_random_effects`：每次均匀抽 1 个块，再在该块内抽 2 个；
   - `two_level_standard`：先从 16 个块中有放回抽 16 个槽位，再从槽位里抽 2 个。
4. 复算预期（`--resamples 10000 --seed 20260930`）：
   `scene_only` **41.7 / 158.3 / 300.0**（P5/P50/P95，`P(S≥130)=0.645`）；
   `block_random_effects` 41.7 / 158.3 / 296.7（0.647）；`two_level_standard` 41.7 / 158.3 / 300.0（0.646）。
   留一法 P50 ∈ [158.3, 166.7]、P95 ∈ [293.3, 308.3]。
5. `per_scenario_j` 只写在 `outputs/` 下（约 75 KB 的 JSON），**不进提交、不占 `artifacts/` 限额**。

## 独立复核发现与修复（补充轮）

独立复核（`notes/verify/VERIFY-REPORT-v2.md`，条目 1–8）发现两处工具级不一致，均已修复：

- **条目 7-③ 输出路径探活**：`--output` 指向不可写路径时，原先会**先跑完全部场景**再在写盘阶段抛未捕获的
  `FileNotFoundError`。现在 `eval_mc.py` / `predict_score.py` 在解析参数后**立即探活**（建目录 + 写探针文件），
  失败即打印一行说明并 **exit 2**，无 traceback、无残留。
- **条目 7-⑤ `--set` 越界值不再静默**：`--set control_gain=1e9` 原先被 CLI 原样接受并写进产物元数据，
  而策略按取值域静默回退到默认值 —— 产物会**记录一个没有生效的配置**，实验归因失真。
  现在 `eval_mc.py` 会构造策略并比较「请求值 vs 生效值」（含未知键），不一致即逐条打印并 **exit 2**。

修复后复测：两类非法输入均 exit 2；合法 `--set` 仍 exit 0；208 局与 `E001-control` 逐位一致；
`rejections` / `scan_hits` 仍为空数组。

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
6. **S3 预测区间**（16 个块 A–P，800+800 场景池）：三模型 `P5/P50/P95` =
   `scene_only` **41.7 / 158.3 / 300.0**、`block_random_effects` 41.7 / 158.3 / 296.7、
   `two_level_standard` 41.7 / 158.3 / 300.0；`P(S≥130)` ≈ **0.645–0.647**。
   块均值 `S(50/50)` ∈ [130.3, 186.3]；块间方差占比 0.9%（池化）/ 1.4%（crossing）/ 3.3%（uniform）；
   留一法 P50 ∈ [158.3, 166.7]、P95 ∈ [293.3, 308.3]。
   假设：私有核验套件与公开套件同构（2 场景 × 2 重复/组）——组织方未公布私有规模，公布后需重跑。
7. **S4 归因**：`eval_mc.py` / `predict_score.py` 的产物 JSON 记录 `tool_sha256`、`seedset_sha256`、
   `entry_sha256` 与 `code_commit`。`entry_sha256` 定位策略、`tool_sha256`/`seedset_sha256` 定位
   **产生数字的工具与种子表**、`code_commit` 定位代码版本。`code_commit` 由调用方注入
   （`acceptance.ps1` 执行 `git rev-parse HEAD`）——**工具内不用 subprocess 调 git**，
   否则会产生人工审核命中项、把 `scan_hits` 从 0 变成非 0。
8. **S1 洁净**：跑完全部工具后提交目录内 `*.pyc` = 0；预检 `working_tree_dirty: false`、
   `file_count = 19 = git ls-files participant/P666`、`total_bytes = 229014` = 这些文件字节和。
7. **F8 移植脚本可复现关键数字**：`real_dynamics_probe.py` 实测 10 步轴向位移 0.2490 m；
   `drive_magnitude_probe.measure_displacement([1,1])` 得 0.3522 m；`geometric_ceiling.py` 得 ≈142.1。
8. **F3 诊断计数器**（`entry.stats`）：6000 次 act 中 `memory_hits=3`（块 A 0、块 B 3），
   `error_count=0`；3 次触发全在 uniform seed 5049 agent 1 step 1–3，且两条路径动作逐位相同 `[1.0, -1.0]`。
9. **F5/F6 机制统计**见 `LOG.md`：6240 次 act 中 66.603% 双分量饱和、96.074% 至少一分量饱和；
   盲走步占 41.35%、全盲 agent-局占 32.83%、t=0 平均可见目标 0.6567。
