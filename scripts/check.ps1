param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"

if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run .\\scripts\\setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($backend)) {
    throw "Torch backend marker is empty. Run .\\scripts\\setup.ps1 again."
}

& uv run --project $repoRoot --frozen --extra $backend python scripts/check_workspace.py
