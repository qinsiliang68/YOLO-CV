$LogDir = "D:\ssh\AI\logs\phase1_node24_dataset_sync_20260623"
Write-Output "PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "node18_sync_node24_missing_normal_splits_20260623|scp -r|scp.exe" } |
    Select-Object ProcessId, Name, CommandLine |
    Format-List
Write-Output "STATUS"
if (Test-Path -LiteralPath (Join-Path $LogDir "status.json")) {
    Get-Content -Raw -LiteralPath (Join-Path $LogDir "status.json")
}
Write-Output "LOG_TAILS"
foreach ($split in @("normal_train", "normal_val_cal", "normal_test")) {
    $out = Join-Path $LogDir "$split.out.log"
    $err = Join-Path $LogDir "$split.err.log"
    Write-Output "--- $split out ---"
    if (Test-Path -LiteralPath $out) { Get-Content -Tail 8 -LiteralPath $out }
    Write-Output "--- $split err ---"
    if (Test-Path -LiteralPath $err) { Get-Content -Tail 8 -LiteralPath $err }
}
