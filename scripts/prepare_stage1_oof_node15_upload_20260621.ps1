param(
  [string]$ArchiveRoot = 'D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch',
  [string]$UploadRoot = 'D:\ssh\AI\upload_ready\stage1_oof_node15_folds_09_10_200epoch_20260621'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

New-Item -ItemType Directory -Force -Path $UploadRoot | Out-Null

$archiveFiles = @(
  'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar',
  'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar.sha256',
  'stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar',
  'stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar.sha256'
)

foreach ($name in $archiveFiles) {
  $src = Join-Path $ArchiveRoot $name
  if (-not (Test-Path -LiteralPath $src)) {
    throw "Missing source file: $src"
  }
  $dst = Join-Path $UploadRoot $name
  if (Test-Path -LiteralPath $dst) {
    Remove-Item -LiteralPath $dst -Force
  }

  if ($name -like '*.tar') {
    New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
  } else {
    Copy-Item -LiteralPath $src -Destination $dst -Force
  }
}

$manifestRoot = Join-Path $ArchiveRoot 'archive_manifests'
$manifestFiles = @(
  'stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541_manifest.txt',
  'stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150_manifest.txt'
)

foreach ($name in $manifestFiles) {
  $src = Join-Path $manifestRoot $name
  if (-not (Test-Path -LiteralPath $src)) {
    throw "Missing manifest file: $src"
  }
  Copy-Item -LiteralPath $src -Destination (Join-Path $UploadRoot $name) -Force
}

$summary = Join-Path $ArchiveRoot 'archive_summary_192_168_100_15_20260621-170438.csv'
if (-not (Test-Path -LiteralPath $summary)) {
  throw "Missing summary file: $summary"
}
Copy-Item -LiteralPath $summary -Destination (Join-Path $UploadRoot 'archive_summary_192_168_100_15_20260621-170438.csv') -Force

$readme = @"
# Upload Package: Stage-1 OOF Node15 Folds 9-10

Created: 2026-06-21
Node: 192.168.100.15
Purpose: Upload-ready backup package for the remaining two completed Stage-1 OOF 200-epoch runs.

Upload these files:

- stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar
- stage1_oof_200epoch_192.168.100.15_fold_08_full_yolo11l_cls_20260620-170541.tar.sha256
- stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar
- stage1_oof_200epoch_192.168.100.15_fold_09_full_yolo11l_cls_20260621-045150.tar.sha256

Fold mapping:

- code fold_08 = human fold 9
- code fold_09 = human fold 10

Each tar includes one completed run directory, the fold manifest material, global OOF fold metadata, and relevant scripts/docs. Each run has:

- results.csv rows: 200
- last_epoch: 200
- weights/best.pt: present
- weights/last.pt: present
- epoch*.pt checkpoints: 200 per fold, 400 total

Do not upload the old duplicate files from the archive root unless specifically needed:

- *.tar.zst
- *.tar.zst.sha256
- archive_summary_192_168_100_15_20260621-170353.csv

Verify after upload by comparing the SHA256 values in the .sha256 files.
"@
Set-Content -LiteralPath (Join-Path $UploadRoot 'README_UPLOAD.txt') -Value $readme -Encoding UTF8

$manifestCsv = Join-Path $UploadRoot 'UPLOAD_MANIFEST.csv'
'relative_path,size_bytes,sha256,modified_at' | Set-Content -LiteralPath $manifestCsv -Encoding UTF8
Get-ChildItem -LiteralPath $UploadRoot -File |
  Sort-Object Name |
  ForEach-Object {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
    ('"{0}","{1}","{2}","{3}"' -f $_.Name, $_.Length, $hash, $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) |
      Add-Content -LiteralPath $manifestCsv -Encoding UTF8
  }

Write-Output "UPLOAD_ROOT=$UploadRoot"
Get-ChildItem -LiteralPath $UploadRoot -File |
  Sort-Object Name |
  Select-Object Name,@{Name='GB';Expression={[math]::Round($_.Length / 1GB, 3)}},Length,LastWriteTime |
  Format-Table -AutoSize

$logicalSize = (Get-ChildItem -LiteralPath $UploadRoot -File | Measure-Object Length -Sum).Sum
Write-Output ("TOTAL_LOGICAL_GB={0}" -f [math]::Round($logicalSize / 1GB, 3))
