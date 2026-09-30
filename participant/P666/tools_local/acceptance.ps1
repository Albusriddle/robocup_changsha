<#
P666 一键验收（F15）。

串起：预检 -> 公开评测 -> 双块 MC -> 鲁棒性自测 -> 范围自查，逐项打印退出码；
任一项失败即返回非零。评审者只需在仓库根目录执行：

    powershell -ExecutionPolicy Bypass -File participant\P666\tools_local\acceptance.ps1

说明：
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
    Write-Host "缺少评测环境解释器: $Py" -ForegroundColor Red
    exit 2
}

$results = @()
function Run-Step {
    param([string]$Name, [string]$Exe, [string[]]$Args, [string]$WorkDir)
    if ($WorkDir -and (Test-Path $WorkDir)) { Remove-Item -Recurse -Force $WorkDir }
    Write-Host ""
    Write-Host ("=== " + $Name + " ===") -ForegroundColor Cyan
    & $Exe @Args 2>&1 | Select-Object -Last 24 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    Write-Host ("-> exit code: " + $code) -ForegroundColor $(if ($code -eq 0) { "Green" } else { "Red" })
    $script:results += [pscustomobject]@{ Step = $Name; ExitCode = $code }
    return $code
}

Write-Host "P666 一键验收" -ForegroundColor Yellow
Write-Host ("仓库根: " + $RepoRoot)
Write-Host ("HEAD  : " + (& git rev-parse HEAD))
Write-Host ("分支  : " + (& git branch --show-current))

# 1) 提交预检
Run-Step -Name "1/5 提交预检 check_submission" -Exe $Py -Args @(
    "scripts\check_submission.py", "--submission", "participant\P666", "--output", (Join-Path $Out "check")
) -WorkDir (Join-Path $Out "check") | Out-Null

# 2) 公开套件评测
Run-Step -Name "2/5 公开评测 evaluate_one" -Exe $Py -Args @(
    "scripts\evaluate_one.py", "--submission", "participant\P666",
    "--suite", "configs\public-suite-v1.yaml", "--seeds", "configs\public-seeds-v1.json",
    "--output", (Join-Path $Out "eval")
) -WorkDir (Join-Path $Out "eval") | Out-Null

# 3) 双块 MC
Run-Step -Name "3/5 双块 MC eval_mc" -Exe $Py -Args @(
    "participant\P666\tools_local\eval_mc.py", "--output", (Join-Path $Out "mc"),
    "--tag", "acceptance", "--seed-block", "both"
) -WorkDir (Join-Path $Out "mc") | Out-Null

# 4) 鲁棒性自测（含 F11 静默降级断言）
Run-Step -Name "4/5 鲁棒性自测 robustness_check" -Exe $Py -Args @(
    "participant\P666\tools_local\robustness_check.py"
) | Out-Null

# 5) 范围自查（只读 git）
Write-Host ""
Write-Host "=== 5/5 范围自查 git diff --name-only master...P666 ===" -ForegroundColor Cyan
$changed = & git diff --name-only master...P666
$changed | ForEach-Object { Write-Host ("   " + $_) }
$outside = $changed | Where-Object { $_ -notlike "participant/P666/*" }
$dirty = & git status --porcelain
if ($outside) {
    Write-Host "-> 越界文件:" -ForegroundColor Red
    $outside | ForEach-Object { Write-Host ("   " + $_) -ForegroundColor Red }
}
if ($dirty) {
    Write-Host "-> 工作区不干净（应为空）:" -ForegroundColor Red
    $dirty | ForEach-Object { Write-Host ("   " + $_) -ForegroundColor Red }
}
$scopeCode = if ($outside -or $dirty) { 1 } else { 0 }
Write-Host ("-> exit code: " + $scopeCode) -ForegroundColor $(if ($scopeCode -eq 0) { "Green" } else { "Red" })
$results += [pscustomobject]@{ Step = "5/5 范围自查"; ExitCode = $scopeCode }

Write-Host ""
Write-Host "=== 汇总 ===" -ForegroundColor Yellow
$results | ForEach-Object { Write-Host ("  " + $_.Step + " -> " + $_.ExitCode) }
$bad = @($results | Where-Object { $_.ExitCode -ne 0 })
if ($bad.Count -gt 0) {
    Write-Host ("验收失败： " + $bad.Count + " 项非零") -ForegroundColor Red
    exit 1
}
Write-Host "验收全绿：5/5 项退出码为 0" -ForegroundColor Green
exit 0
