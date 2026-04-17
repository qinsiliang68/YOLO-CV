param()

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location -LiteralPath $repoRoot

$uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$config = Join-Path $repoRoot "YOLOv11\configs\runtime\stage1_formal_gate_bucket_pilot_machine_b_cq3_smoke.json"

$command = @(
    $uv,
    "run",
    "python",
    "scripts\stage1_formal_gate_bucket_pilot.py",
    "--config",
    $config
)

Write-Host "[run] $($command -join ' ')"
& $command[0] $command[1..($command.Length - 1)]
