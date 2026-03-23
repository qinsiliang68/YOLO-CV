param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

if (-not (Test-Path $venv)) {
    & $Python -m venv $venv
}

$py = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Python executable not found in virtual environment: $py"
}

& $py -m pip install --upgrade pip setuptools wheel

$requirements = Join-Path $root "requirements.txt"
if (Test-Path $requirements) {
    & $py -m pip install -r $requirements
}

& $py -m pip install -e $root

Write-Host "YOLOv11 environment is ready."
Write-Host "Next steps:"
Write-Host "  1. Edit configs/datasets/custom_detect.yaml"
Write-Host "  2. Put your dataset under datasets/"
Write-Host "  3. Run scripts/train.ps1"
