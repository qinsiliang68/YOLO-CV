param(
    [Parameter(Mandatory = $true)][int]$NodeIndex,
    [string]$NodeLabel = $env:COMPUTERNAME
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Script = Join-Path $Repo "scripts\ops\run_phase1_hn_rn_formal_node_20260623.ps1"
$PhaseRoot = "D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623"
$TaskName = "YOLO_CV_phase1_hn_rn_${NodeLabel}_node${NodeIndex}_20260623"
$CombinedLog = Join-Path $PhaseRoot ("{0}_formal_node{1}_task.log" -f $NodeLabel, $NodeIndex)

if (-not (Test-Path -LiteralPath $Script)) {
    throw "Missing formal run script: $Script"
}

New-Item -ItemType Directory -Force -Path $PhaseRoot | Out-Null

$existing = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "run_phase1_hn_rn_formal_node_20260623|run_stage1_phase1_hn_rn_pipeline_20260623|train_stage1_cls_sweep" -and
            $_.CommandLine -notmatch "launch_phase1_hn_rn_formal_node_20260623"
        }
)
if ($existing.Count -gt 0) {
    $desc = ($existing | ForEach-Object { "{0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " ") }) -join "; "
    throw "Existing formal/training process found: $desc"
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
Remove-Item -LiteralPath $CombinedLog -ErrorAction SilentlyContinue

$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ArgumentText = '-NoProfile -ExecutionPolicy Bypass -Command "& ''{0}'' -NodeIndex {1} -NodeLabel ''{2}'' *> ''{3}''"' -f $Script, $NodeIndex, $NodeLabel, $CombinedLog
$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $ArgumentText -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Description "YOLO-CV phase1 HN/RN formal node-index $NodeIndex" | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Output "SCHEDULED_FORMAL_NODE task=$TaskName node_label=$NodeLabel node_index=$NodeIndex log=$CombinedLog"
