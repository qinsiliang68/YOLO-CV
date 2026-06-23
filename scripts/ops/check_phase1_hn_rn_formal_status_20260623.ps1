param(
    [Parameter(Mandatory = $true)][int]$NodeIndex,
    [string]$NodeLabel = $env:COMPUTERNAME
)

$ErrorActionPreference = "Continue"

$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_20260623"
$EvalRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623\eval"
$TaskName = "YOLO_CV_phase1_hn_rn_${NodeLabel}_node${NodeIndex}_20260623"
$CombinedLog = Join-Path $PhaseRoot ("{0}_formal_node{1}_task.log" -f $NodeLabel, $NodeIndex)
$Plan = @{
    1 = @("HN-01", "RN-01", "HN-02", "RN-02")
    2 = @("HN-03", "RN-03", "HN-04", "RN-04")
    3 = @("HN-05", "RN-05", "HN-06", "RN-06")
    4 = @("HN-07", "RN-07", "HN-08", "RN-08")
    5 = @("HN-09", "RN-09", "HN-10", "RN-10")
    6 = @("HN-11", "RN-11", "HN-12", "RN-12")
    7 = @("HN-13", "RN-13", "HN-14", "RN-14")
    8 = @("HN-15", "RN-15", "HN-16", "RN-16")
    9 = @("HN-17", "RN-17", "HN-18", "RN-18")
    10 = @("HN-19", "RN-19", "HN-20", "RN-20")
}

Write-Output "time=$(Get-Date -Format s)"
Write-Output "node_label=$NodeLabel"
Write-Output "node_index=$NodeIndex"
Write-Output "assignment=$($Plan[$NodeIndex] -join ',')"
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

$procs = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "validate_stage1_phase1_hn_rn_manifests_20260623|run_stage1_phase1_hn_rn_pipeline_20260623|train_stage1_cls_sweep|evaluate_stage1_cls_gate" -and
            $_.CommandLine -notmatch "check_phase1_hn_rn_formal_status"
        }
)
Write-Output "matching_proc_count=$($procs.Count)"
$procs | Select-Object -First 10 | ForEach-Object {
    Write-Output ("proc={0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " "))
}

foreach ($runId in $Plan[$NodeIndex]) {
    $runRoot = Join-Path $RunsRoot $runId
    $evalRunRoot = Join-Path $EvalRoot $runId
    $summary = Join-Path (Join-Path $PhaseRoot "pipeline_summaries") ($runId + ".json")
    $log = Join-Path (Join-Path $PhaseRoot "pipeline_logs") ($runId + ".log")
    Write-Output "run=$runId"
    Write-Output "  summary_exists=$(Test-Path -LiteralPath $summary)"
    if (Test-Path -LiteralPath $summary) {
        Write-Output "  summary_tail=$((Get-Content -LiteralPath $summary -Raw) -replace '\s+', ' ')"
    }
    Write-Output "  log_exists=$(Test-Path -LiteralPath $log)"
    if (Test-Path -LiteralPath $log) {
        Write-Output "  log_tail=$((Get-Content -LiteralPath $log -Tail 8) -join ' | ')"
    }
    if (Test-Path -LiteralPath $runRoot) {
        Get-ChildItem -LiteralPath $runRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @("best.pt", "last.pt") } |
            ForEach-Object { Write-Output "  weight=$($_.FullName)" }
    }
    if (Test-Path -LiteralPath $evalRunRoot) {
        Get-ChildItem -LiteralPath $evalRunRoot -Recurse -File -Filter "*predictions*.csv" -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "  predictions=$($_.FullName)" }
    }
}
