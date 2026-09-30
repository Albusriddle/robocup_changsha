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
Write-Output ("HEAD  : " + (& git rev-parse HEAD))
Write-Output ("分支  : " + (& git branch --show-current))

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
    "--tag", "acceptance", "--seed-block", "both"
) -WorkDir (Join-Path $Out "mc")

Run-Step -Name "4/5 鲁棒性自测 robustness_check" -Exe $Py -CmdArgs @(
    "participant\P666\tools_local\robustness_check.py"
)

Write-Output ""
Write-Output "=== 5/5 范围自查 git diff --name-only master...P666 ==="
$changed = @(& git diff --name-only master...P666)
$changed | ForEach-Object { Write-Output ("   " + $_) }
$outside = @($changed | Where-Object { $_ -notlike "participant/P666/*" })
$dirty = @(& git status --porcelain)
if ($outside.Count -gt 0) {
    Write-Output "-> 越界文件:"
    $outside | ForEach-Object { Write-Output ("   " + $_) }
}
if ($dirty.Count -gt 0) {
    Write-Output "-> 工作区不干净（应为空）:"
    $dirty | ForEach-Object { Write-Output ("   " + $_) }
}
$scopeCode = if (($outside.Count -gt 0) -or ($dirty.Count -gt 0)) { 1 } else { 0 }
Write-Output ("-> exit code: " + $scopeCode)
$results += [pscustomobject]@{ Step = "5/5 范围自查"; ExitCode = $scopeCode }

Write-Output ""
Write-Output "=== 汇总 ==="
$results | ForEach-Object { Write-Output ("  " + $_.Step + " -> " + $_.ExitCode) }
$bad = @($results | Where-Object { $_.ExitCode -ne 0 })
if ($bad.Count -gt 0) {
    Write-Output ("验收失败：" + $bad.Count + " 项非零")
    exit 1
}
Write-Output "验收全绿：5/5 项退出码为 0"
exit 0
