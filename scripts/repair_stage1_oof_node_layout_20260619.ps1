param(
    [string]$RepoRoot = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV",
    [string]$StorageRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Get-IsReparse {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    return (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Get-ReparseTarget {
    param([Parameter(Mandatory=$true)][string]$Path)
    $item = Get-Item -LiteralPath $Path -Force
    $target = $null
    try {
        $target = @($item.Target)[0]
    } catch {
        $target = $null
    }
    if ([string]::IsNullOrWhiteSpace($target)) {
        throw "Could not read reparse target for $Path"
    }
    return $target
}

function Assert-PathUnder {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Root
    )
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root)
    if (-not $fullRoot.EndsWith("\")) { $fullRoot += "\" }
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing operation outside root. path=$fullPath root=$fullRoot"
    }
}

function Remove-JunctionOnly {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Get-IsReparse -Path $Path)) {
        throw "Refusing to remove non-reparse path as junction: $Path"
    }
    if ($DryRun) {
        Write-Step "[dry-run] remove junction only: $Path"
        return
    }
    [System.IO.Directory]::Delete($Path, $false)
}

function Invoke-RobocopyMove {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing source for $Label`: $Source"
    }
    if ($DryRun) {
        Write-Step "[dry-run] robocopy /MOVE $Label`: $Source -> $Destination"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $script:StorageRootFinal "logs") | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $safeLabel = $Label -replace "[^A-Za-z0-9_.-]", "_"
    $log = Join-Path $script:StorageRootFinal "logs\robocopy_${safeLabel}_${stamp}.log"
    Write-Step "robocopy /MOVE $Label`: $Source -> $Destination"
    & robocopy $Source $Destination /E /MOVE /MT:32 /R:1 /W:1 /XJ /COPY:DAT /DCOPY:DAT /NP /NFL /NDL /TEE /LOG:$log
    $code = $LASTEXITCODE
    Write-Step "robocopy exit for $Label`: $code log=$log"
    if ($code -ge 8) {
        throw "Robocopy failed for $Label with exit code $code"
    }
}

function Remove-DirectoryAfterMove {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$AllowedRoot
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Assert-PathUnder -Path $Path -Root $AllowedRoot
    if ($DryRun) {
        Write-Step "[dry-run] remove moved source directory: $Path"
        return
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Ensure-JunctionToStorage {
    param(
        [Parameter(Mandatory=$true)][string]$LinkPath,
        [Parameter(Mandatory=$true)][string]$TargetPath,
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][string]$AllowedRemoveRoot
    )

    New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null

    if (Test-Path -LiteralPath $LinkPath) {
        if (Get-IsReparse -Path $LinkPath) {
            $currentTarget = Get-ReparseTarget -Path $LinkPath
            if ($currentTarget.TrimEnd("\") -ieq $TargetPath.TrimEnd("\")) {
                Write-Step "$Label already junctioned to $TargetPath"
                return
            }
            Write-Step "$Label is a junction to $currentTarget; replacing with $TargetPath"
            Remove-JunctionOnly -Path $LinkPath
        } else {
            Write-Step "$Label is a real directory; moving contents to $TargetPath"
            Invoke-RobocopyMove -Source $LinkPath -Destination $TargetPath -Label $Label
            Remove-DirectoryAfterMove -Path $LinkPath -AllowedRoot $AllowedRemoveRoot
        }
    }

    if ($DryRun) {
        Write-Step "[dry-run] create junction: $LinkPath -> $TargetPath"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LinkPath) | Out-Null
    cmd /c mklink /J "$LinkPath" "$TargetPath" | Out-Host
}

function Ensure-DatasetOnC {
    param(
        [Parameter(Mandatory=$true)][string]$DatasetParent,
        [Parameter(Mandatory=$true)][string]$DatasetName
    )

    $datasetDest = Join-Path $DatasetParent $DatasetName
    if (Test-Path -LiteralPath $DatasetParent) {
        if (Get-IsReparse -Path $DatasetParent) {
            $targetParent = Get-ReparseTarget -Path $DatasetParent
            $datasetSource = Join-Path $targetParent $DatasetName
            Write-Step "dataset parent is junction: $DatasetParent -> $targetParent"
            Remove-JunctionOnly -Path $DatasetParent
            if (-not $DryRun) {
                New-Item -ItemType Directory -Force -Path $DatasetParent | Out-Null
            }
            if (Test-Path -LiteralPath $datasetSource) {
                Invoke-RobocopyMove -Source $datasetSource -Destination $datasetDest -Label "dataset_$DatasetName"
                Remove-DirectoryAfterMove -Path $datasetSource -AllowedRoot $targetParent
            } elseif (Test-Path -LiteralPath $datasetDest) {
                Write-Step "dataset already exists on C: $datasetDest"
            } else {
                throw "Dataset source missing after junction removal: $datasetSource"
            }
        } else {
            Write-Step "dataset parent is already a real directory: $DatasetParent"
        }
    } else {
        Write-Step "creating real C dataset parent: $DatasetParent"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Force -Path $DatasetParent | Out-Null
        }
    }

    $manifest = Join-Path $datasetDest "manifests\train_manifest.csv"
    if (-not $DryRun -and -not (Test-Path -LiteralPath $manifest)) {
        throw "Dataset verification failed. Missing manifest: $manifest"
    }
}

function Ensure-RealCWorkdir {
    param([Parameter(Mandatory=$true)][string]$WorkdirPath)

    if (Test-Path -LiteralPath $WorkdirPath) {
        if (Get-IsReparse -Path $WorkdirPath) {
            $target = Get-ReparseTarget -Path $WorkdirPath
            Write-Step "active workdir is junction: $WorkdirPath -> $target"
            Write-Step "leaving old generated workdir target in place and replacing active path with real C directory"
            Remove-JunctionOnly -Path $WorkdirPath
        } else {
            Write-Step "active workdir already real: $WorkdirPath"
        }
    }
    if ($DryRun) {
        Write-Step "[dry-run] ensure real C workdir: $WorkdirPath"
        return
    }
    New-Item -ItemType Directory -Force -Path $WorkdirPath | Out-Null
}

function Detect-Node {
    $runsRoot = Join-Path $RepoRoot "YOLOv11\runs\stage1_oof_10fold"
    if (Test-Path -LiteralPath (Join-Path $runsRoot "fold_04")) { return "13" }
    if (Test-Path -LiteralPath (Join-Path $runsRoot "fold_00")) { return "18" }
    throw "Could not detect node from OOF run folders under $runsRoot"
}

$aiRoot = "C:\Users\ASUS\Desktop\ssh\AI"
$node = Detect-Node
if ([string]::IsNullOrWhiteSpace($StorageRoot)) {
    if ($node -eq "13") {
        $StorageRoot = "F:\ssh\AI"
    } elseif ($node -eq "18") {
        $StorageRoot = "D:\ssh\AI"
    } else {
        throw "Missing StorageRoot for node $node"
    }
}
$script:StorageRootFinal = $StorageRoot

Write-Step "node=$node repo=$RepoRoot storage=$StorageRoot dry_run=$DryRun"
if ($StorageRoot.Substring(0, 2).ToUpperInvariant() -eq "C:") {
    throw "StorageRoot must not be on C: $StorageRoot"
}

$repoRuns = Join-Path $RepoRoot "YOLOv11\runs"
$repoRunsTarget = Join-Path $StorageRoot "runs\YOLOv11"
Ensure-JunctionToStorage -LinkPath $repoRuns -TargetPath $repoRunsTarget -Label "repo_YOLOv11_runs" -AllowedRemoveRoot (Join-Path $RepoRoot "YOLOv11")

Ensure-JunctionToStorage -LinkPath (Join-Path $aiRoot "runs") -TargetPath (Join-Path $StorageRoot "runs") -Label "ai_runs" -AllowedRemoveRoot $aiRoot
Ensure-JunctionToStorage -LinkPath (Join-Path $aiRoot "logs") -TargetPath (Join-Path $StorageRoot "logs") -Label "ai_logs" -AllowedRemoveRoot $aiRoot
Ensure-JunctionToStorage -LinkPath (Join-Path $aiRoot "artifacts") -TargetPath (Join-Path $StorageRoot "artifacts") -Label "ai_artifacts" -AllowedRemoveRoot $aiRoot
Ensure-JunctionToStorage -LinkPath (Join-Path $aiRoot "checkpoint_archive") -TargetPath (Join-Path $StorageRoot "checkpoint_archive") -Label "ai_checkpoint_archive" -AllowedRemoveRoot $aiRoot

$datasetParent = Join-Path $aiRoot "datasets"
Ensure-DatasetOnC -DatasetParent $datasetParent -DatasetName "final_sewerml_dataset"

if ($node -eq "13") {
    $workName = "stage1_oof_workdir"
} else {
    $workName = "stage1_oof_workdir_cdrive"
}
$workdirPath = Join-Path (Join-Path $RepoRoot "data") $workName
Ensure-RealCWorkdir -WorkdirPath $workdirPath

$commonWorkdirs = Join-Path $aiRoot "workdirs"
if (-not (Test-Path -LiteralPath $commonWorkdirs)) {
    Write-Step "creating real C common workdirs path: $commonWorkdirs"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $commonWorkdirs | Out-Null
    }
}

Write-Step "layout repair completed"
