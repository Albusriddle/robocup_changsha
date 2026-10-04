<#
P666 一键验收（F15）。

串起：预检 -> 公开评测 -> 双块 MC -> 鲁棒性自测 -> 范围自查，逐项打印退出码；
任一项失败即返回非零。评审者只需在仓库根目录执行：

    powershell -ExecutionPolicy Bypass -File participant\P666\tools_local\acceptance.ps1

说明：
- 输出统一用 Write-Output（便于被上层管道/日志捕获）。
- 使用仓库根的 .venv\Scripts\python.exe（Python 3.12 + 官方依赖锁）。
- 每次评测前清空目标输出目录（官方评测器要求输出目录为空，否则 OUTPUT_DIRECTORY_NOT_EMPTY）。
- 只读操作，不修改任何提交文件；范围自查用只读 git diff。
#>

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Out = Join-Path $RepoRoot "outputs\P666\acceptance"

if (-not (Test-Path $Py)) {
    Write-Output ("缺少评测环境解释器: " + $Py)
    exit 2
}

$results = @()

function Run-Step {
    param([string]$Name, [string]$Exe, [string[]]$CmdArgs, [string]$WorkDir)
    if ($WorkDir -and (Test-Path $WorkDir)) { Remove-Item -Recurse -Force $WorkDir }
    Write-Output ""
    Write-Output ("=== " + $Name + " ===")
    & $Exe @CmdArgs 2>&1 | Select-Object -Last 22 | ForEach-Object { Write-Output $_ }
    $code = $LASTEXITCODE
    Write-Output ("-> exit code: " + $code)
    $script:results += [pscustomobject]@{ Step = $Name; ExitCode = $code }
}

Write-Output "P666 一键验收"
Write-Output ("仓库根: " + $RepoRoot)
$Commit = (& git rev-parse HEAD)
Write-Output ("HEAD  : " + $Commit)
Write-Output ("分支  : " + (& git branch --show-current))
if ($Commit) {
    $Commit = $Commit.Trim()
    Write-Output ("S4: 将把 code_commit=" + $Commit + " 注入工具产物（.ps1 不进 AST 审计，故由这里取 git 值）")
} else {
    Write-Output "S4: 未能取得 commit（工具产物将记录 code_commit=null）"
}

Run-Step -Name "1/5 提交预检 check_submission" -Exe $Py -CmdArgs @(
    "scripts\check_submission.py", "--submission", "participant\P666", "--output", (Join-Path $Out "check")
) -WorkDir (Join-Path $Out "check")

Run-Step -Name "2/5 公开评测 evaluate_one" -Exe $Py -CmdArgs @(
    "scripts\evaluate_one.py", "--submission", "participant\P666",
    "--suite", "configs\public-suite-v1.yaml", "--seeds", "configs\public-seeds-v1.json",
    "--output", (Join-Path $Out "eval")
) -WorkDir (Join-Path $Out "eval")

Run-Step -Name "3/5 双块 MC eval_mc" -Exe $Py -CmdArgs @(
    "participant\P666\tools_local\eval_mc.py", "--output", (Join-Path $Out "mc"),
    "--tag", "acceptance", "--seed-block", "both", "--code-commit", $Commit
) -WorkDir (Join-Path $Out "mc")

Run-Step -Name "4/5 鲁棒性自测 robustness_check" -Exe $Py -CmdArgs @(
    "participant\P666\tools_local\robustness_check.py"
)

Write-Output ""
Write-Output "=== 5/5 范围与洁净自查 ==="
Write-Output "--- git diff --name-only master...P666 ---"
$changed = @(& git diff --name-only master...P666)
$changed | ForEach-Object { Write-Output ("   " + $_) }
$outside = @($changed | Where-Object { $_ -notlike "participant/P666/*" })

# S1-a：提交目录内不得残留 Python 字节码缓存。
# 官方预检把"被忽略的文件"也算作未提交内容（coverage_bench/submission.py:144-161），
# 一份 __pycache__ 就会让 working_tree_dirty=true、并把 file_count/total_bytes 虚增。
$pyc = @(Get-ChildItem -Recurse -Force (Join-Path $RepoRoot "participant\P666") -Filter *.pyc -ErrorAction SilentlyContinue)
Write-Output ("--- 提交目录内 *.pyc 数量（应为 0）: " + $pyc.Count + " ---")
$pyc | ForEach-Object { Write-Output ("   " + $_.FullName) }

# S1-b：使用与官方同语义的洁净检查（--ignored，并限定在提交目录），
# 而不是 git status --porcelain —— 后者会忽略被 ignore 的文件，永远报绿。
Write-Output "--- git status --porcelain --ignored -- participant/P666（应为空）---"
$dirty = @(& git status --porcelain --ignored -- participant/P666)
$dirty | ForEach-Object { Write-Output ("   " + $_) }

if ($outside.Count -gt 0) {
    Write-Output "-> 越界文件:"
    $outside | ForEach-Object { Write-Output ("   " + $_) }
}
$scopeCode = if (($outside.Count -gt 0) -or ($dirty.Count -gt 0) -or ($pyc.Count -gt 0)) { 1 } else { 0 }
Write-Output ("-> exit code: " + $scopeCode)
$results += [pscustomobject]@{ Step = "5/5 范围与洁净自查"; ExitCode = $scopeCode }

Write-Output ""
# ---- 材料哈希自检（W3）：抽取材料里所有 64 位十六进制串，逐个校验 ----
# 目的：自动拦住"文档写着旧哈希/写错哈希"这类矛盾（本轮曾出现 §4 头部沿用旧 entry.py 摘要）。
Write-Output ""
Write-Output "=== 材料哈希自检（REPORT/LOG/experiments/submission）==="
$known = @{}
foreach ($rel in @("participant\P666\entry.py", "participant\P666\artifacts\rule-config.json")) {
    $full = Join-Path $RepoRoot $rel
    $known[(Get-FileHash -Algorithm SHA256 -LiteralPath $full).Hash.ToLower()] = "current:$rel"
}
# 历史内容哈希（由 notes\verify\hash_identity.py 以 git cat-file blob 二进制安全复算；见 REPORT.md §13）
$historical = @{
    "3e66c74b41f3c9706ec186a999d0cf3ceaaf48bee81d803334c9cfd7cc708ca3" = "entry.py @ V1（rebase 前 aa54761 / 后 663aba2）"
    "205bcc16a648398fe5f16b4ce59af86502d7953ecf70571ae40ee2be2a8f9275" = "entry.py @ e18a9b1（F3 计数器）"
    "fe9e604908c8c4d2ff404ea1256cb0e107982e5d7e51df20de0cca0684e63f26" = "entry.py @ defd0d2（S 轮末）"
    "bb55d0464b9374f7fd3e1d976ebadc8c9e1bef0e2e170a507d883210d36fd9bd" = "rule-config.json @ E008 前（449 B）"
}
$badHash = @()
$seen = 0
foreach ($rel in @("participant\P666\REPORT.md", "participant\P666\LOG.md",
                   "participant\P666\experiments.csv", "participant\P666\submission.yaml")) {
    $text = Get-Content -LiteralPath (Join-Path $RepoRoot $rel) -Raw
    foreach ($m in [regex]::Matches($text, '\b[0-9a-f]{64}\b')) {
        $seen++
        $h = $m.Value
        if (-not ($known.ContainsKey($h) -or $historical.ContainsKey($h))) { $badHash += ($rel + " " + $h) }
    }
}
Write-Output ("--- 扫描到 64 位哈希 " + $seen + " 处；已知集合 " + ($known.Count + $historical.Count) + " 个 ---")
if ($badHash.Count -gt 0) {
    Write-Output "-> 未登记/不一致的哈希:"
    $badHash | ForEach-Object { Write-Output ("   " + $_) }
}
$hashCode = if ($badHash.Count -gt 0) { 1 } else { 0 }
Write-Output ("-> exit code: " + $hashCode)
$results += [pscustomobject]@{ Step = "材料哈希自检"; ExitCode = $hashCode }
Write-Output "=== 汇总 ==="
$results | ForEach-Object { Write-Output ("  " + $_.Step + " -> " + $_.ExitCode) }
$bad = @($results | Where-Object { $_.ExitCode -ne 0 })
if ($bad.Count -gt 0) {
    Write-Output ("验收失败：" + $bad.Count + " 项非零")
    exit 1
}
Write-Output "验收全绿：5/5 项退出码为 0"
exit 0
