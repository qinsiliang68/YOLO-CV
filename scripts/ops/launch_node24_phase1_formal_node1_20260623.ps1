$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Script = Join-Path $Repo "scripts\ops\node24_phase1_formal_node1_20260623.ps1"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$Stdout = Join-Path $PhaseRoot "node24_formal_node1_stdout.log"
$Stderr = Join-Path $PhaseRoot "node24_formal_node1_stderr.log"

if (-not (Test-Path -LiteralPath $Script)) {
    throw "Missing formal script: $Script"
}

New-Item -ItemType Directory -Force -Path $PhaseRoot | Out-Null

$existing = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "node24_phase1_formal_node1_20260623|run_stage1_phase1_hn_rn_pipeline_20260623|train_stage1_cls_sweep" -and
            $_.CommandLine -notmatch "launch_node24_phase1_formal_node1_20260623"
        }
)
if ($existing.Count -gt 0) {
    $desc = ($existing | ForEach-Object { "{0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " ") }) -join "; "
    throw "Existing formal/training process found: $desc"
}

$proc = Start-Process `
    -WindowStyle Hidden `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Script) `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

Write-Output "LAUNCHED_NODE24_FORMAL_NODE1 pid=$($proc.Id) stdout=$Stdout stderr=$Stderr"
