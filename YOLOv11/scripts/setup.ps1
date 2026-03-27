param(
    [ValidateSet("cpu", "cu126", "cu128")]
    [string]$Backend = "cu128",
    [string]$Python = "3.11"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& (Join-Path $repoRoot "scripts\setup.ps1") -Backend $Backend -Python $Python
