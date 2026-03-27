param(
    [ValidateSet("sewerml", "cam-review", "train-metrics")]
    [string]$Mode = "sewerml",
    [string]$AnnotationsDir = "",
    [string]$ReviewCsv = "",
    [string]$ResultsCsv = "",
    [string]$OutputDir = "",
    [string]$Output = "",
    [string[]]$Splits = @("Train", "Val"),
    [string]$Title = "",
    [int]$Dpi = 220
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendFile = Join-Path $repoRoot ".uv-torch-backend"
if (-not (Test-Path $backendFile)) {
    throw "Torch backend marker not found. Run .\\scripts\\setup.ps1 first."
}

$backend = (Get-Content $backendFile -Raw).Trim()
$args = @("run", "--project", $repoRoot, "--frozen", "--extra", $backend, "python", "scripts/plot_research_figures.py", $Mode, "--dpi", $Dpi)

if ($Mode -eq "sewerml") {
    if ($AnnotationsDir) { $args += @("--annotations-dir", $AnnotationsDir) }
    if ($OutputDir) { $args += @("--output-dir", $OutputDir) }
    $cleanSplits = @()
    foreach ($split in $Splits) {
        if (-not [string]::IsNullOrWhiteSpace($split)) {
            $cleanSplits += $split
        }
    }
    if ($cleanSplits.Count -gt 0) {
        $args += "--splits"
        $args += $cleanSplits
    }
}
elseif ($Mode -eq "cam-review") {
    if ($ReviewCsv) { $args += @("--review-csv", $ReviewCsv) }
    if ($OutputDir) { $args += @("--output-dir", $OutputDir) }
}
else {
    if (-not $ResultsCsv) {
        throw "ResultsCsv is required when Mode=train-metrics."
    }
    $args += @("--results-csv", $ResultsCsv)
    if ($Output) { $args += @("--output", $Output) }
    if ($Title) { $args += @("--title", $Title) }
}

& uv @args
