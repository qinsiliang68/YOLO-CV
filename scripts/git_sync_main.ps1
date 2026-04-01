$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "[git] repo: $repoRoot"
Write-Host "[git] fetch origin"
git fetch origin
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] switch main"
git switch main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] reset local main to origin/main"
git reset --hard origin/main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] synced to origin/main"
git status --short --branch
