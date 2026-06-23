param(
    [Parameter(Mandatory = $true)][string]$NodeLabel,
    [string]$RunId = "HN-01"
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Script = Join-Path $Repo "scripts\ops\smoke_phase1_hn_rn_node_20260623.ps1"
$SmokeRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_${NodeLabel}_20260623_1ep"
$TaskName = "YOLO_CV_phase1_hn_rn_smoke_${NodeLabel}_20260623"
$CombinedLog = Join-Path $SmokeRoot "${NodeLabel}_smoke_task.log"

if (-not (Test-Path -LiteralPath $Script)) {
    throw "Missing smoke script: $Script"
}

New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

$existing = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "smoke_phase1_hn_rn_node_20260623|run_stage1_phase1_hn_rn_pipeline_20260623|train_stage1_cls_sweep" -and
            $_.CommandLine -notmatch "launch_phase1_hn_rn_smoke_node_20260623"
        }
)
if ($existing.Count -gt 0) {
    $desc = ($existing | ForEach-Object { "{0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " ") }) -join "; "
    throw "Existing smoke/training process found: $desc"
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
Remove-Item -LiteralPath $CombinedLog -ErrorAction SilentlyContinue

$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ArgumentText = '-NoProfile -ExecutionPolicy Bypass -Command "& ''{0}'' -NodeLabel ''{1}'' -RunId ''{2}'' *> ''{3}''"' -f $Script, $NodeLabel, $RunId, $CombinedLog
$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $ArgumentText -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Description "YOLO-CV phase1 HN/RN smoke for $NodeLabel" | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Output "SCHEDULED_SMOKE_NODE task=$TaskName node_label=$NodeLabel run_id=$RunId log=$CombinedLog"
