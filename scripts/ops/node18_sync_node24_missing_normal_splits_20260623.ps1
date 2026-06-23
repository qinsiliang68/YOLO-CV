$ErrorActionPreference = "Stop"

$SourceImages = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset\Det\images"
$TargetParent = "ASUS@192.168.100.24:C:/Users/ASUS/Desktop/ssh/AI/datasets/final_sewerml_dataset/Det/images/"
$Key = "C:\Users\ASUS\.ssh\node_to_node_ed25519"
$LogDir = "D:\ssh\AI\logs\phase1_node24_dataset_sync_20260623"
$StatusPath = Join-Path $LogDir "status.json"
$Splits = @("normal_train", "normal_val_cal", "normal_test")

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Count-Files {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return -1 }
    return (Get-ChildItem -LiteralPath $Path -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

$status = [ordered]@{
    started_at = (Get-Date).ToString("s")
    source_images = $SourceImages
    target_parent = $TargetParent
    splits = $Splits
    results = @()
    complete = $false
    failed = $false
}

foreach ($split in $Splits) {
    $source = Join-Path $SourceImages $split
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing source split: $source"
    }
    $beforeCount = Count-Files $source
    $outLog = Join-Path $LogDir "$split.out.log"
    $errLog = Join-Path $LogDir "$split.err.log"
    "START split=$split source_count=$beforeCount time=$(Get-Date -Format s)" | Add-Content -LiteralPath $outLog -Encoding utf8
    & scp -r -p -q -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL $source $TargetParent 1>>$outLog 2>>$errLog
    $exit = $LASTEXITCODE
    "END split=$split exit=$exit time=$(Get-Date -Format s)" | Add-Content -LiteralPath $outLog -Encoding utf8
    $status.results += [ordered]@{
        split = $split
        source_count = $beforeCount
        exit_code = $exit
        out_log = $outLog
        err_log = $errLog
    }
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding utf8
    if ($exit -ne 0) {
        $status.failed = $true
        $status.complete = $false
        $status.ended_at = (Get-Date).ToString("s")
        $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding utf8
        throw "scp failed for $split with exit $exit"
    }
}

$status.complete = $true
$status.failed = $false
$status.ended_at = (Get-Date).ToString("s")
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding utf8
Write-Output "SYNC_COMPLETE status=$StatusPath"
