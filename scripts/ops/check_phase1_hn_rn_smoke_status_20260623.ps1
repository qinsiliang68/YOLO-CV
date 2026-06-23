param(
    [Parameter(Mandatory = $true)][string]$NodeLabel,
    [string]$RunId = "HN-01"
)

$ErrorActionPreference = "Continue"

$SmokeRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$EvalRoot = Join-Path $SmokeRoot "eval"
$TaskName = "YOLO_CV_phase1_hn_rn_smoke_${NodeLabel}_20260623"
$CombinedLog = Join-Path $SmokeRoot "${NodeLabel}_smoke_task.log"

Write-Output "time=$(Get-Date -Format s)"
Write-Output "node_label=$NodeLabel"
Write-Output "run_id=$RunId"
Write-Output "gpu=$((& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null) -join ';')"
Write-Output "compute_procs=$((& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null) -join ';')"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    $task = Get-ScheduledTask -TaskName $TaskName
    $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Output "task_state=$($task.State)"
    Write-Output "task_last_result=$($taskInfo.LastTaskResult)"
    Write-Output "task_last_run=$($taskInfo.LastRunTime)"
}
else {
    Write-Output "task_state=missing"
}
Write-Output "combined_log_exists=$(Test-Path -LiteralPath $CombinedLog)"
if (Test-Path -LiteralPath $CombinedLog) {
    Write-Output "combined_log_tail=$((Get-Content -LiteralPath $CombinedLog -Tail 12) -join ' | ')"
}

$summary = Join-Path (Join-Path $SmokeRoot "pipeline_summaries") ($RunId + ".json")
$verify = Join-Path $SmokeRoot ("verification_{0}.json" -f $RunId)
Write-Output "summary_exists=$(Test-Path -LiteralPath $summary)"
if (Test-Path -LiteralPath $summary) {
    Write-Output "summary_tail=$((Get-Content -LiteralPath $summary -Raw) -replace '\s+', ' ')"
}
Write-Output "verification_exists=$(Test-Path -LiteralPath $verify)"
if (Test-Path -LiteralPath $verify) {
    Write-Output "verification_tail=$((Get-Content -LiteralPath $verify -Raw) -replace '\s+', ' ')"
}
if (Test-Path -LiteralPath $RunsRoot) {
    Get-ChildItem -LiteralPath $RunsRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("best.pt", "last.pt") } |
        ForEach-Object { Write-Output "weight=$($_.FullName)" }
}
if (Test-Path -LiteralPath $EvalRoot) {
    Get-ChildItem -LiteralPath $EvalRoot -Recurse -File -Filter "*predictions*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Output "predictions=$($_.FullName)" }
}
