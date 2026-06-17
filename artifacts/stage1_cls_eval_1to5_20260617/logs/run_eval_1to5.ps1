$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = "D:\ssh\AI\projects\YOLO-CV"
$DatasetRoot = "D:\data\final_sewerml_dataset"
$YoloRoot = "D:\ssh\AI\projects\YOLO-CV\YOLOv11"
$OutputRoot = "D:\ssh\AI\runs\stage1_cls_eval_1to5"
$LogRoot = "D:\ssh\AI\logs\stage1_cls_eval_1to5"
$UvPath = "C:\Users\ASUS\.local\bin\uv.exe"
$Batch = 128
$Device = "0"
$TargetRecall = 0.995
$DeploymentDefectPrevalence = 0.10
$Splits = "val_cal,val_op,test"

function NowIso {
    return (Get-Date).ToString("s")
}

function Write-Json {
    param($Object, [string] $Path)
    $Object | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $Path
}

function Count-CsvRows {
    param([string] $Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return -1
    }
    $n = 0
    [System.IO.File]::ReadLines($Path) | Select-Object -Skip 1 | ForEach-Object { $n++ }
    return $n
}

function Count-Files {
    param([string] $Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return -1
    }
    return (Get-ChildItem -File -LiteralPath $Path -ErrorAction SilentlyContinue | Measure-Object).Count
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

$statusPath = Join-Path $LogRoot "eval_1to5_status.json"
$summaryPath = Join-Path $LogRoot "eval_1to5_summary.csv"

$required = @(
    @{Name = "val_cal"; DefectManifest = "val_cal_manifest.csv"; NormalManifest = "normal_val_cal_manifest.csv"; DefectDir = "val_cal"; NormalDir = "normal_val_cal"},
    @{Name = "val_op"; DefectManifest = "val_op_manifest.csv"; NormalManifest = "normal_val_op_manifest.csv"; DefectDir = "val_op"; NormalDir = "normal_val_op"},
    @{Name = "test"; DefectManifest = "test_manifest.csv"; NormalManifest = "normal_test_manifest.csv"; DefectDir = "test"; NormalDir = "normal_test"}
)

$checks = foreach ($r in $required) {
    $dm = Join-Path $DatasetRoot "manifests\$($r.DefectManifest)"
    $nm = Join-Path $DatasetRoot "manifests\$($r.NormalManifest)"
    $dd = Join-Path $DatasetRoot "Det\images\$($r.DefectDir)"
    $nd = Join-Path $DatasetRoot "Det\images\$($r.NormalDir)"
    [ordered]@{
        split = $r.Name
        defect_manifest_rows = Count-CsvRows $dm
        normal_manifest_rows = Count-CsvRows $nm
        defect_image_files = Count-Files $dd
        normal_image_files = Count-Files $nd
    }
}

$bad = @($checks | Where-Object {
    $_.defect_manifest_rows -lt 0 -or
    $_.normal_manifest_rows -lt 0 -or
    $_.defect_image_files -lt 0 -or
    $_.normal_image_files -lt 0 -or
    $_.defect_manifest_rows -ne $_.defect_image_files -or
    $_.normal_manifest_rows -ne $_.normal_image_files
})

if ($bad.Count -gt 0) {
    $payload = [ordered]@{
        started_at = NowIso
        status = "blocked_dataset_incomplete"
        dataset_root = $DatasetRoot
        checks = $checks
    }
    Write-Json $payload $statusPath
    throw "Dataset incomplete; see $statusPath"
}

$jobs = @(
    @{model = "n"; run_name = "eval_1to5_full_yolo11n_cls_20260614-190411_best"; weights = "D:\ssh\AI\runs\stage1_cls_sweep\full_yolo11n_cls_20260614-190411\weights\best.pt"},
    @{model = "s"; run_name = "eval_1to5_full_yolo11s_cls_20260615-063433_best"; weights = "D:\ssh\AI\runs\stage1_cls_sweep\full_yolo11s_cls_20260615-063433\weights\best.pt"},
    @{model = "m"; run_name = "eval_1to5_full_yolo11m_cls_20260615-180049_best"; weights = "D:\ssh\AI\runs\stage1_cls_sweep\full_yolo11m_cls_20260615-180049\weights\best.pt"},
    @{model = "l"; run_name = "eval_1to5_full_yolo11l_cls_20260615-123305_best"; weights = "D:\ssh\AI\weights\imported\yolo11l_best.pt"},
    @{model = "x"; run_name = "eval_1to5_full_yolo11x_cls_20260614-185818_best"; weights = "D:\ssh\AI\weights\imported\yolo11x_best.pt"}
)

$status = [ordered]@{
    started_at = NowIso
    finished_at = $null
    status = "running"
    project_root = $ProjectRoot
    dataset_root = $DatasetRoot
    yolo_root = $YoloRoot
    output_root = $OutputRoot
    splits = $Splits
    batch = $Batch
    device = $Device
    target_recall = $TargetRecall
    deployment_defect_prevalence = $DeploymentDefectPrevalence
    checks = $checks
    jobs = @()
}
Write-Json $status $statusPath

Set-Location $ProjectRoot

foreach ($job in $jobs) {
    $log = Join-Path $LogRoot "$($job.run_name).log"
    $entry = [ordered]@{
        model = $job.model
        run_name = $job.run_name
        weights = $job.weights
        log = $log
        started_at = NowIso
        finished_at = $null
        status = "running"
        exit_code = $null
    }
    $status.jobs += $entry
    Write-Json $status $statusPath
    "[$(NowIso)] start model=$($job.model) run=$($job.run_name)" | Set-Content -Encoding UTF8 -LiteralPath $log

    if (!(Test-Path -LiteralPath $job.weights)) {
        "weights not found: $($job.weights)" | Add-Content -Encoding UTF8 -LiteralPath $log
        $entry.status = "failed"
        $entry.exit_code = 9001
        $entry.finished_at = NowIso
        Write-Json $status $statusPath
        continue
    }

    $args = @(
        "run", "--no-sync", "python", "scripts\evaluate_stage1_cls_gate.py",
        "--weights", $job.weights,
        "--dataset-root", $DatasetRoot,
        "--yolo-root", $YoloRoot,
        "--output-root", $OutputRoot,
        "--run-name", $job.run_name,
        "--splits", $Splits,
        "--batch", [string] $Batch,
        "--device", $Device,
        "--target-recall", [string] $TargetRecall,
        "--deployment-defect-prevalence", [string] $DeploymentDefectPrevalence,
        "--exist-ok"
    )

    try {
        & $UvPath @args >> $log 2>&1
        $code = $LASTEXITCODE
    } catch {
        $_ | Out-String | Add-Content -Encoding UTF8 -LiteralPath $log
        $code = 9002
    }

    $entry.exit_code = $code
    $entry.finished_at = NowIso
    $entry.status = if ($code -eq 0) { "completed" } else { "failed" }
    Write-Json $status $statusPath
}

$metricRows = @()
foreach ($job in $jobs) {
    $metrics = Join-Path $OutputRoot "$($job.run_name)\metrics_at_selected_threshold.csv"
    if (Test-Path -LiteralPath $metrics) {
        Import-Csv -LiteralPath $metrics | ForEach-Object {
            $_ | Add-Member -NotePropertyName model -NotePropertyValue $job.model -Force
            $_ | Add-Member -NotePropertyName run_name -NotePropertyValue $job.run_name -Force
            $metricRows += $_
        }
    }
}

if ($metricRows.Count -gt 0) {
    $metricRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $summaryPath
}

$status.finished_at = NowIso
$status.status = if (@($status.jobs | Where-Object { $_.status -ne "completed" }).Count -eq 0) {
    "completed"
} else {
    "completed_with_failures"
}
$status.summary_csv = $summaryPath
Write-Json $status $statusPath
