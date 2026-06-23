$PidToCheck = 13956
$LogDir = "D:\ssh\AI\logs\phase1_node24_smoke_20260623"
$SmokeRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2"
$EvalRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2\eval"

Write-Output "PROCESS"
Get-Process -Id $PidToCheck -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-List
Write-Output "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "stage1_phase1_hn_rn|train_stage1|evaluate_stage1|ultralytics|yolo" } |
    Select-Object ProcessId,Name,CommandLine |
    Format-List
Write-Output "GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
Write-Output "LOG_OUT_TAIL"
if (Test-Path -LiteralPath (Join-Path $LogDir "smoke.out.log")) {
    Get-Content -Tail 60 -LiteralPath (Join-Path $LogDir "smoke.out.log")
}
Write-Output "LOG_ERR_TAIL"
if (Test-Path -LiteralPath (Join-Path $LogDir "smoke.err.log")) {
    Get-Content -Tail 40 -LiteralPath (Join-Path $LogDir "smoke.err.log")
}
Write-Output "OUTPUTS"
foreach ($path in @(
    (Join-Path $SmokeRoot "pipeline_summaries\HN-01.json"),
    (Join-Path $SmokeRoot "last_pipeline_batch.json"),
    (Join-Path $RunsRoot "HN-01"),
    (Join-Path $EvalRoot "HN-01\eval_HN-01_best")
)) {
    $exists = Test-Path -LiteralPath $path
    Write-Output "$path exists=$exists"
}
