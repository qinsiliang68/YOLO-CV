param(
    [string]$BaseDir = "C:\GitHub\YOLO-CV\data\sewerml",
    [string]$OutputDir = "C:\GitHub\YOLO-CV\YOLOv11\datasets\sewerml_cls6_train3000",
    [int]$TrainPerClass = 450,
    [int]$ValPerClass = 50,
    [int]$Seed = 42,
    [ValidateSet("hardlink", "copy")]
    [string]$Mode = "hardlink",
    [switch]$Clean
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $PSScriptRoot "extract_sewerml_cls6_train3000.py"

$args = @(
    "run",
    "--project", $repoRoot,
    "python",
    $scriptPath,
    "--base-dir", $BaseDir,
    "--output-dir", $OutputDir,
    "--train-per-class", $TrainPerClass,
    "--val-per-class", $ValPerClass,
    "--seed", $Seed,
    "--mode", $Mode
)

if ($Clean) {
    $args += "--clean"
}

uv @args
