param(
  [Parameter(Mandatory=$true)][string]$Node,
  [Parameter(Mandatory=$true)][string]$Repo,
  [Parameter(Mandatory=$true)][string]$RunsRoot,
  [Parameter(Mandatory=$true)][string]$CheckpointRoot,
  [Parameter(Mandatory=$true)][string]$OutRoot,
  [Parameter(Mandatory=$true)][int[]]$Folds
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$tmpRoot = Join-Path $OutRoot '_tmp'
$manifestRoot = Join-Path $OutRoot 'archive_manifests'
New-Item -ItemType Directory -Force -Path $OutRoot, $tmpRoot, $manifestRoot | Out-Null
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot

$summary = Join-Path $OutRoot ('archive_summary_' + $Node.Replace('.', '_') + '_' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.csv')
'node,fold,archive_path,archive_size_gb,sha256,run_path,run_size_gb,run_file_count,run_epoch_pt_count,external_epoch_pt_count,results_rows,last_epoch,best_pt,last_pt,created_at' |
  Set-Content -LiteralPath $summary -Encoding UTF8

$scriptRelPaths = @(
  'scripts\build_stage1_oof_folds.py',
  'scripts\run_stage1_oof_folds_20260617.py',
  'scripts\continue_stage1_oof_node_20260619.py',
  'scripts\train_stage1_cls_sweep.py',
  'scripts\repair_stage1_oof_node_layout_20260619.ps1',
  'scripts\relocate_stage1_code_env_20260619.ps1',
  'scripts\inspect_stage1_oof_node_layout_20260619.ps1',
  'scripts\validate_stage1_oof_continue_20260619.ps1',
  'docs\stage1_oof_10fold.md',
  'tests\test_build_stage1_oof_folds.py',
  'tests\test_train_stage1_manifest_dir.py'
)
$codeFiles = foreach ($rel in $scriptRelPaths) {
  $p = Join-Path $Repo $rel
  if (Test-Path $p) { $p }
}

$artifactRoot = Join-Path $Repo 'artifacts\stage1_oof_folds_10fold_20260617'
$globalArtifacts = foreach ($rel in @('fold_summary.csv', 'fold_jobs.csv', 'group_summary.csv', 'metadata.json', 'train_oof_assignments.csv')) {
  $p = Join-Path $artifactRoot $rel
  if (Test-Path $p) { $p }
}

$gitBranch = ''
$gitCommit = ''
$gitStatus = @()
try {
  $gitBranch = (& git -C $Repo rev-parse --abbrev-ref HEAD).Trim()
  $gitCommit = (& git -C $Repo rev-parse --short HEAD).Trim()
  $gitStatus = & git -C $Repo status --short
} catch {
  $gitStatus = @('git_status_unavailable=' + $_.Exception.Message)
}

foreach ($fold in $Folds) {
  $foldName = ('fold_{0:d2}' -f $fold)
  $foldPath = Join-Path $RunsRoot $foldName
  $run = Get-ChildItem $foldPath -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $run) {
    throw "Missing run directory for $foldName under $foldPath"
  }

  $runFiles = Get-ChildItem $run.FullName -Recurse -File
  $runSize = ($runFiles | Measure-Object Length -Sum).Sum
  $runEpochCount = ($runFiles | Where-Object { $_.Name -match '^epoch\d+\.pt$' }).Count

  $externalCheckpointDir = Join-Path (Join-Path $CheckpointRoot $foldName) $run.Name
  $externalEpochCount = 0
  $sources = New-Object System.Collections.Generic.List[string]
  $sources.Add($run.FullName)
  if (Test-Path $externalCheckpointDir) {
    $externalEpochCount = (Get-ChildItem $externalCheckpointDir -Recurse -File -Filter 'epoch*.pt').Count
    $sources.Add($externalCheckpointDir)
  }

  $foldArtifact = Join-Path (Join-Path $artifactRoot 'folds') $foldName
  if (Test-Path $foldArtifact) {
    $sources.Add($foldArtifact)
  }
  foreach ($p in $globalArtifacts) { $sources.Add($p) }
  foreach ($p in $codeFiles) { $sources.Add($p) }

  $resultsPath = Join-Path $run.FullName 'results.csv'
  $rows = ''
  $lastEpoch = ''
  if (Test-Path $resultsPath) {
    $csv = Import-Csv $resultsPath
    $rows = $csv.Count
    $lastRow = $csv | Select-Object -Last 1
    $epochName = $lastRow.PSObject.Properties.Name | Where-Object { $_ -match '^\s*epoch\s*$' } | Select-Object -First 1
    if ($epochName) { $lastEpoch = $lastRow.$epochName }
  }

  $bestPath = Join-Path $run.FullName 'weights\best.pt'
  $lastPath = Join-Path $run.FullName 'weights\last.pt'
  $archive = Join-Path $OutRoot ("stage1_oof_200epoch_${Node}_${foldName}_$($run.Name).tar")
  if (Test-Path $archive) {
    throw "Archive already exists: $archive"
  }

  $manifest = Join-Path $manifestRoot ("stage1_oof_200epoch_${Node}_${foldName}_$($run.Name)_manifest.txt")
  $codeHashLines = foreach ($p in $codeFiles) {
    $h = (Get-FileHash -Algorithm SHA256 $p).Hash
    "code_sha256=$h path=$p"
  }
  $sourceLines = foreach ($p in $sources) { "source=$p" }
  $statusLines = foreach ($s in $gitStatus) { "git_status=$s" }

  @(
    "node=$Node",
    "fold=$foldName",
    "run=$($run.FullName)",
    "run_size_gb=$([math]::Round($runSize / 1GB, 3))",
    "run_file_count=$($runFiles.Count)",
    "run_epoch_pt_count=$runEpochCount",
    "external_checkpoint_dir=$externalCheckpointDir",
    "external_epoch_pt_count=$externalEpochCount",
    "results_rows=$rows",
    "last_epoch=$lastEpoch",
    "best_pt_exists=$(Test-Path $bestPath)",
    "last_pt_exists=$(Test-Path $lastPath)",
    "git_branch=$gitBranch",
    "git_commit=$gitCommit",
    "archive_path=$archive",
    "created_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    $statusLines,
    $codeHashLines,
    $sourceLines
  ) | Set-Content -LiteralPath $manifest -Encoding UTF8

  $tarSources = @($manifest) + $sources.ToArray()
  Write-Output "ARCHIVE_START $foldName $archive"
  & tar -cf $archive @tarSources
  if ($LASTEXITCODE -ne 0) {
    throw "tar failed for $foldName with exit code $LASTEXITCODE"
  }

  $hash = (Get-FileHash -Algorithm SHA256 $archive).Hash
  $shaPath = $archive + '.sha256'
  ($hash + '  ' + (Split-Path $archive -Leaf)) | Set-Content -LiteralPath $shaPath -Encoding ASCII
  $archiveSize = (Get-Item $archive).Length

  ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}' -f
    $Node,
    $foldName,
    $archive,
    [math]::Round($archiveSize / 1GB, 3),
    $hash,
    $run.FullName,
    [math]::Round($runSize / 1GB, 3),
    $runFiles.Count,
    $runEpochCount,
    $externalEpochCount,
    $rows,
    $lastEpoch,
    (Test-Path $bestPath),
    (Test-Path $lastPath),
    (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) |
      Add-Content -LiteralPath $summary -Encoding UTF8

  Write-Output ("ARCHIVE_DONE {0} SIZE_GB={1} SHA256={2}" -f $foldName, [math]::Round($archiveSize / 1GB, 3), $hash)
}

Write-Output "SUMMARY=$summary"
foreach ($driveName in @('C', 'D', 'E', 'F')) {
  $drive = Get-PSDrive -Name $driveName -PSProvider FileSystem -ErrorAction SilentlyContinue
  if ($drive) {
    Write-Output ("DISK_{0}_FREE_GB={1}" -f $driveName, [math]::Round($drive.Free / 1GB, 2))
  }
}
