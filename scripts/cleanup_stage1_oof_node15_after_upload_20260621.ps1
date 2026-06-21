param(
  [string]$ArchiveRoot = 'D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch',
  [string]$UploadRoot = 'D:\ssh\AI\upload_ready\stage1_oof_node15_folds_09_10_200epoch_20260621',
  [string]$RunRootBase = 'D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$bigFiles = @(
  (Join-Path $UploadRoot 'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar'),
  (Join-Path $UploadRoot 'stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar'),
  (Join-Path $UploadRoot 'stage1_oof_200epoch_192.168.100.15_fold_9_full_yolo11l_cls_20260620-170541.tar'),
  (Join-Path $UploadRoot 'stage1_oof_200epoch_192.168.100.15_fold_10_full_yolo11l_cls_20260621-045150.tar'),
  (Join-Path $ArchiveRoot 'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar'),
  (Join-Path $ArchiveRoot 'stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar'),
  (Join-Path $ArchiveRoot 'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar.zst')
)

$runDirs = @(
  (Join-Path $RunRootBase 'fold_08'),
  (Join-Path $RunRootBase 'fold_09')
)

function Assert-AllowedFile {
  param([Parameter(Mandatory=$true)][string]$Path)
  $full = [IO.Path]::GetFullPath($Path)
  $archiveFull = [IO.Path]::GetFullPath($ArchiveRoot)
  $uploadFull = [IO.Path]::GetFullPath($UploadRoot)
  if (-not ($full.StartsWith($archiveFull, [StringComparison]::OrdinalIgnoreCase) -or $full.StartsWith($uploadFull, [StringComparison]::OrdinalIgnoreCase))) {
    throw "Refusing file outside allowed roots: $full"
  }
  if (-not ($full.EndsWith('.tar', [StringComparison]::OrdinalIgnoreCase) -or $full.EndsWith('.tar.zst', [StringComparison]::OrdinalIgnoreCase))) {
    throw "Refusing non-archive file: $full"
  }
}

function Assert-AllowedDir {
  param([Parameter(Mandatory=$true)][string]$Path)
  $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $base = [IO.Path]::GetFullPath($RunRootBase).TrimEnd('\')
  if (-not $full.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing directory outside run root: $full"
  }
  if (-not ($full.EndsWith('\fold_08', [StringComparison]::OrdinalIgnoreCase) -or $full.EndsWith('\fold_09', [StringComparison]::OrdinalIgnoreCase))) {
    throw "Refusing unexpected run directory: $full"
  }
}

Write-Output 'BEFORE_DISK='
Get-PSDrive -PSProvider FileSystem |
  Where-Object { $_.Name -in @('C', 'D') } |
  Select-Object Name,@{Name='FreeGB';Expression={[math]::Round($_.Free / 1GB, 2)}},@{Name='UsedGB';Expression={[math]::Round($_.Used / 1GB, 2)}} |
  Format-Table -AutoSize

$deleted = New-Object System.Collections.Generic.List[object]

foreach ($path in $bigFiles) {
  Assert-AllowedFile -Path $path
  if (Test-Path -LiteralPath $path) {
    $item = Get-Item -LiteralPath $path
    $size = $item.Length
    Remove-Item -LiteralPath $path -Force
    $deleted.Add([PSCustomObject]@{
      Type = 'file'
      Path = $path
      GB = [math]::Round($size / 1GB, 3)
      Files = ''
    })
  }
}

foreach ($path in $runDirs) {
  Assert-AllowedDir -Path $path
  if (Test-Path -LiteralPath $path) {
    $files = Get-ChildItem -LiteralPath $path -Recurse -File
    $size = ($files | Measure-Object Length -Sum).Sum
    $count = $files.Count
    Remove-Item -LiteralPath $path -Recurse -Force
    $deleted.Add([PSCustomObject]@{
      Type = 'dir'
      Path = $path
      GB = [math]::Round($size / 1GB, 3)
      Files = $count
    })
  }
}

Write-Output 'DELETED='
$deleted | Format-Table -AutoSize

Write-Output 'AFTER_DISK='
Get-PSDrive -PSProvider FileSystem |
  Where-Object { $_.Name -in @('C', 'D') } |
  Select-Object Name,@{Name='FreeGB';Expression={[math]::Round($_.Free / 1GB, 2)}},@{Name='UsedGB';Expression={[math]::Round($_.Used / 1GB, 2)}} |
  Format-Table -AutoSize

Write-Output 'REMAINING_BIG_ARCHIVES='
Get-ChildItem -LiteralPath $ArchiveRoot -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '192\.168\.100\.15.*\.(tar|tar\.zst)$' } |
  Select-Object Name,@{Name='GB';Expression={[math]::Round($_.Length / 1GB, 3)}} |
  Format-Table -AutoSize

Write-Output 'REMAINING_RUN_DIRS='
$remainingRunDirs = foreach ($path in $runDirs) {
  [PSCustomObject]@{ Path = $path; Exists = (Test-Path -LiteralPath $path) }
}
$remainingRunDirs | Format-Table -AutoSize
