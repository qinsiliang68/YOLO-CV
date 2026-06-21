param(
  [string]$ArchiveRoot = 'D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch',
  [string]$UploadRoot = 'D:\ssh\AI\upload_ready\stage1_oof_node15_folds_09_10_200epoch_20260621'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

New-Item -ItemType Directory -Force -Path $UploadRoot | Out-Null

function Read-Sha256Sidecar {
  param([Parameter(Mandatory=$true)][string]$Path)

  $text = Get-Content -LiteralPath $Path -Raw
  $match = [regex]::Match($text, '(?i)\b[a-f0-9]{64}\b')
  if (-not $match.Success) {
    throw "Invalid SHA256 sidecar: $Path"
  }
  return $match.Value.ToUpperInvariant()
}

function Assert-Sha256SidecarMatches {
  param(
    [Parameter(Mandatory=$true)][string]$FilePath,
    [Parameter(Mandatory=$true)][string]$SidecarPath
  )

  if (-not (Test-Path -LiteralPath $SidecarPath)) {
    throw "Missing SHA256 sidecar: $SidecarPath"
  }
  $expected = Read-Sha256Sidecar -Path $SidecarPath
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $FilePath).Hash.ToUpperInvariant()
  if ($actual -ne $expected) {
    throw "SHA256 mismatch: $FilePath expected=$expected actual=$actual"
  }
}

function Move-StagedFile {
  param(
    [Parameter(Mandatory=$true)][string]$TempPath,
    [Parameter(Mandatory=$true)][string]$Destination
  )

  if (Test-Path -LiteralPath $Destination) {
    $name = Split-Path -Leaf $Destination
    $backup = Join-Path (Split-Path -Parent $Destination) ('.backup-{0}-{1}-{2}' -f $PID, ([guid]::NewGuid().ToString('N')), $name)
    Move-Item -LiteralPath $Destination -Destination $backup
    try {
      Move-Item -LiteralPath $TempPath -Destination $Destination
      Remove-Item -LiteralPath $backup -Force
    } catch {
      if ((-not (Test-Path -LiteralPath $Destination)) -and (Test-Path -LiteralPath $backup)) {
        Move-Item -LiteralPath $backup -Destination $Destination
      }
      throw
    }
    return
  }

  if (-not (Test-Path -LiteralPath $Destination)) {
    Move-Item -LiteralPath $TempPath -Destination $Destination
  }
}

function New-StagedHardLink {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination
  )

  $name = Split-Path -Leaf $Destination
  $tmpDst = Join-Path (Split-Path -Parent $Destination) ('.stage-{0}-{1}-{2}' -f $PID, ([guid]::NewGuid().ToString('N')), $name)
  try {
    New-Item -ItemType HardLink -Path $tmpDst -Target $Source | Out-Null
    Move-StagedFile -TempPath $tmpDst -Destination $Destination
  } catch {
    if (Test-Path -LiteralPath $tmpDst) {
      Remove-Item -LiteralPath $tmpDst -Force
    }
    throw
  }
}

function Copy-StagedFile {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination
  )

  $name = Split-Path -Leaf $Destination
  $tmpDst = Join-Path (Split-Path -Parent $Destination) ('.stage-{0}-{1}-{2}' -f $PID, ([guid]::NewGuid().ToString('N')), $name)
  try {
    Copy-Item -LiteralPath $Source -Destination $tmpDst -Force
    Move-StagedFile -TempPath $tmpDst -Destination $Destination
  } catch {
    if (Test-Path -LiteralPath $tmpDst) {
      Remove-Item -LiteralPath $tmpDst -Force
    }
    throw
  }
}

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

  if ($name -like '*.tar') {
    Assert-Sha256SidecarMatches -FilePath $src -SidecarPath (Join-Path $ArchiveRoot "$name.sha256")
  }

  $dst = Join-Path $UploadRoot $name

  if ($name -like '*.tar') {
    New-StagedHardLink -Source $src -Destination $dst
  } else {
    Copy-StagedFile -Source $src -Destination $dst
  }
}

foreach ($name in $archiveFiles) {
  if ($name -like '*.tar') {
    Assert-Sha256SidecarMatches -FilePath (Join-Path $UploadRoot $name) -SidecarPath (Join-Path $UploadRoot "$name.sha256")
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

Verify after upload by comparing the remote SHA256 values with the .sha256 files.
Only after remote SHA256 verification passes, create REMOTE_UPLOAD_VERIFIED.txt in this upload root with:

REMOTE_UPLOAD_SHA256_VERIFIED=YES
"@
Set-Content -LiteralPath (Join-Path $UploadRoot 'README_UPLOAD.txt') -Value $readme -Encoding UTF8

$remoteVerifyTemplate = @"
REMOTE_UPLOAD_SHA256_VERIFIED=NO
REMOTE_PATH=
VERIFIED_AT=
VERIFIED_BY=
"@
Set-Content -LiteralPath (Join-Path $UploadRoot 'REMOTE_UPLOAD_VERIFIED_TEMPLATE.txt') -Value $remoteVerifyTemplate -Encoding UTF8

$localVerified = @"
LOCAL_PACKAGE_SHA256_VERIFIED=YES
VERIFIED_AT=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
"@
Set-Content -LiteralPath (Join-Path $UploadRoot 'LOCAL_PACKAGE_VERIFIED.txt') -Value $localVerified -Encoding UTF8

$manifestCsv = Join-Path $UploadRoot 'UPLOAD_MANIFEST.csv'
'relative_path,size_bytes,sha256,modified_at' | Set-Content -LiteralPath $manifestCsv -Encoding UTF8
Get-ChildItem -LiteralPath $UploadRoot -File |
  Where-Object { $_.Name -ne 'UPLOAD_MANIFEST.csv' } |
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
