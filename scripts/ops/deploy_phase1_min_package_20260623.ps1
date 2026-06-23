$ErrorActionPreference = "Stop"

$TarPath = "C:\Users\ASUS\Desktop\ssh\AI\tmp\yolo_cv_phase1_min_08cc6bb.tar"
$Target = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Commit = "08cc6bb"

if (-not (Test-Path -LiteralPath $TarPath)) {
    throw "Missing tar package: $TarPath"
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null
tar -xf $TarPath -C $Target
if ($LASTEXITCODE -ne 0) {
    throw "tar extraction failed with exit code $LASTEXITCODE"
}

Set-Content -LiteralPath (Join-Path $Target "DEPLOYED_COMMIT.txt") -Encoding ascii -Value $Commit

$required = @(
    "scripts\build_stage1_phase1_hn_rn_manifests_20260623.py",
    "scripts\run_stage1_phase1_hn_rn_pipeline_20260623.py",
    "scripts\validate_stage1_phase1_hn_rn_manifests_20260623.py",
    "scripts\train_stage1_cls_sweep.py",
    "scripts\evaluate_stage1_cls_gate.py",
    "YOLOv11\ultralytics\__init__.py",
    "artifacts\stage1_oof_predictions_calop_20260621\merged_10fold_20260622\oof_predictions_merged.csv"
)

foreach ($rel in $required) {
    $path = Join-Path $Target $rel
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required deployed file: $path"
    }
}

Write-Output "DEPLOY_OK target=$Target commit=$Commit"
