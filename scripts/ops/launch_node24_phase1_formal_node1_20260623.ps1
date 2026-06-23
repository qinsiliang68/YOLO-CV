$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Script = Join-Path $Repo "scripts\ops\node24_phase1_formal_node1_20260623.ps1"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$Stdout = Join-Path $PhaseRoot "node24_formal_node1_stdout.log"
$Stderr = Join-Path $PhaseRoot "node24_formal_node1_stderr.log"
$CombinedLog = Join-Path $PhaseRoot "node24_formal_node1_task.log"
$TaskName = "YOLO_CV_node24_phase1_formal_node1_20260623"

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

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
Remove-Item -LiteralPath $Stdout, $Stderr, $CombinedLog -ErrorAction SilentlyContinue

$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ArgumentText = '-NoProfile -ExecutionPolicy Bypass -Command "& ''{0}'' *> ''{1}''"' -f $Script, $CombinedLog
$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $ArgumentText -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Description "YOLO-CV node24 phase1 HN/RN formal node-index 1" | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Output "SCHEDULED_NODE24_FORMAL_NODE1 task=$TaskName log=$CombinedLog args=$ArgumentText"
