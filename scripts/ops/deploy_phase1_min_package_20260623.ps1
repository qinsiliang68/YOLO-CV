param(
    [string]$TarPath = "C:\Users\ASUS\Desktop\ssh\AI\tmp\yolo_cv_phase1_min.tar",
    [string]$Target = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV",
    [Parameter(Mandatory = $true)][string]$Commit
)

$ErrorActionPreference = "Stop"

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
    "scripts\verify_stage1_phase1_hn_rn_outputs_20260623.py",
    "scripts\ops\run_phase1_hn_rn_formal_node_20260623.ps1",
    "scripts\ops\launch_phase1_hn_rn_formal_node_20260623.ps1",
    "scripts\ops\check_phase1_hn_rn_formal_status_20260623.ps1",
    "scripts\ops\smoke_phase1_hn_rn_node_20260623.ps1",
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
