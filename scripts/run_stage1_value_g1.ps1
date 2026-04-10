param(
    [switch]$DryRun,
    [switch]$Rerun,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location -LiteralPath $repoRoot

$command = @("uv", "run", "main.py", "--task", "stage1_formal_gate_value_g1")
if ($DryRun) { $command += "--dry-run" }
if ($Rerun) { $command += "--rerun" }
if ($PreflightOnly) { $command += "--preflight-only" }

Write-Host "[run] $($command -join ' ')"
& $command[0] $command[1..($command.Length - 1)]
