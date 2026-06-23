$ErrorActionPreference = "Continue"

$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$RunsRoot = "D:\ssh\AI\runs\stage1_phase1_hn_rn_20260623"
$EvalRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623\eval"

Write-Output "time=$(Get-Date -Format s)"
Write-Output "gpu=$((& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null) -join ';')"
Write-Output "compute_procs=$((& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null) -join ';')"

$procs = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "run_stage1_phase1_hn_rn_pipeline_20260623|train_stage1_cls_sweep|evaluate_stage1_cls_gate" -and
            $_.CommandLine -notmatch "check_node24_formal_status"
        }
)
Write-Output "matching_proc_count=$($procs.Count)"
$procs | Select-Object -First 10 | ForEach-Object {
    Write-Output ("proc={0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " "))
}

foreach ($runId in @("HN-01", "RN-01", "HN-02", "RN-02")) {
    $runRoot = Join-Path $RunsRoot $runId
    $evalRoot = Join-Path $EvalRoot $runId
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
        Get-ChildItem -LiteralPath $runRoot -Recurse -File -Include best.pt,last.pt -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "  weight=$($_.FullName)" }
    }
    if (Test-Path -LiteralPath $evalRoot) {
        Get-ChildItem -LiteralPath $evalRoot -Recurse -File -Filter "*predictions*.csv" -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Output "  predictions=$($_.FullName)" }
    }
}
