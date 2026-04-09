param(
    [ValidateSet("cu128")]
    [string]$Backend = "cu128",
    [string]$Python = "3.11"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not on PATH."
}

Set-Content -Path $backendFile -Value $Backend -NoNewline -Encoding ascii

& uv sync --project $repoRoot --python $Python --extra $Backend --frozen
& uv run --project $repoRoot --frozen --extra $Backend python scripts/check_workspace.py --create-dirs

Write-Host "Workspace environment is ready."
Write-Host "Selected torch backend: $Backend"
Write-Host "Next steps:"
Write-Host "  1. Move datasets into the local-only paths described in research/training_machine_runbook.md"
Write-Host "  2. Run .\\scripts\\check.ps1 to verify the folder layout"
Write-Host "  3. Start the current training task with uv run main.py or use .\\YOLOv11\\scripts\\train.ps1 for direct runs"
