$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

param(
    [string]$Message = ""
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "results: update materials and results $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Host "[git] repo: $repoRoot"
Write-Host "[git] switch main"
git switch main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] stage result directories"
git add research/materials research/results
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$staged = git diff --cached --name-only
if ([string]::IsNullOrWhiteSpace(($staged | Out-String))) {
    Write-Host "[git] no staged result changes"
    git status --short --branch
    exit 0
}

Write-Host "[git] commit"
git commit -m $Message
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] rebase onto latest origin/main"
git pull --rebase origin main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] push main"
git push origin main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[git] results pushed to origin/main"
git status --short --branch
