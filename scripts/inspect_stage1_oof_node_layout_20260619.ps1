param(
    [string]$RepoRoot = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
)

$ErrorActionPreference = "Stop"

function Get-LinkInfo {
    param([Parameter(Mandatory=$true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject][ordered]@{
            path = $Path
            exists = $false
        }
    }

    $item = Get-Item -LiteralPath $Path -Force
    $isReparse = (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
    $target = $null
    try {
        if ($null -ne $item.Target) {
            $target = @($item.Target)
        }
    } catch {
        $target = $null
    }

    return [pscustomobject][ordered]@{
        path = $Path
        exists = $true
        full_name = $item.FullName
        attributes = $item.Attributes.ToString()
        reparse = $isReparse
        link_type = $item.LinkType
        target = $target
    }
}

function Get-ReparseAncestors {
    param([Parameter(Mandatory=$true)][string]$Path)

    $result = @()
    $current = $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $info = Get-LinkInfo -Path $current
            if ($info.exists -and $info.reparse) {
                $result += $info
            }
        }
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) {
            break
        }
        $current = $parent
    }
    return $result
}

function Get-RunStatus {
    param([Parameter(Mandatory=$true)][string]$RunsRoot)

    $rows = @()
    if (-not (Test-Path -LiteralPath $RunsRoot)) {
        return @()
    }

    Get-ChildItem -LiteralPath $RunsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "fold_*" } |
        Sort-Object Name |
        ForEach-Object {
            $foldDir = $_
            Get-ChildItem -LiteralPath $foldDir.FullName -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne "_logs" } |
                Sort-Object LastWriteTime |
                ForEach-Object {
                    $runDir = $_
                    $results = Join-Path $runDir.FullName "results.csv"
                    $lastEpoch = $null
                    if (Test-Path -LiteralPath $results) {
                        try {
                            $lines = Get-Content -LiteralPath $results -Tail 5 -ErrorAction Stop
                            foreach ($line in $lines) {
                                if ($line -match "^\s*([0-9]+)") {
                                    $lastEpoch = [int]$Matches[1]
                                }
                            }
                        } catch {
                            $lastEpoch = $null
                        }
                    }
                    $weightsDir = Join-Path $runDir.FullName "weights"
                    $epochCount = 0
                    if (Test-Path -LiteralPath $weightsDir) {
                        $epochCount = @(Get-ChildItem -LiteralPath $weightsDir -Filter "epoch*.pt" -File -ErrorAction SilentlyContinue).Count
                    }
                    $rows += [pscustomobject][ordered]@{
                        fold = $foldDir.Name
                        run = $runDir.Name
                        path = $runDir.FullName
                        last_write_time = $runDir.LastWriteTime.ToString("s")
                        result_epoch = $lastEpoch
                        best_pt = Test-Path -LiteralPath (Join-Path $weightsDir "best.pt")
                        last_pt = Test-Path -LiteralPath (Join-Path $weightsDir "last.pt")
                        epoch_pt_count = $epochCount
                    }
                }
        }
    return $rows
}

$aiRoot = "C:\Users\ASUS\Desktop\ssh\AI"
$paths = @(
    $aiRoot,
    (Join-Path $aiRoot "datasets"),
    (Join-Path $aiRoot "datasets\final_sewerml_dataset"),
    (Join-Path $aiRoot "workdirs"),
    (Join-Path $aiRoot "runs"),
    (Join-Path $aiRoot "logs"),
    (Join-Path $aiRoot "artifacts"),
    (Join-Path $aiRoot "checkpoint_archive"),
    (Join-Path $aiRoot "repos"),
    $RepoRoot,
    (Join-Path $RepoRoot "data\stage1_oof_workdir"),
    (Join-Path $RepoRoot "data\stage1_oof_workdir_cdrive"),
    (Join-Path $RepoRoot "YOLOv11\runs"),
    (Join-Path $RepoRoot "YOLOv11\runs\stage1_oof_10fold")
)

$procRows = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match "^(python|uv|cmd|powershell).*\.exe$" -and
        $_.CommandLine -match "continue_stage1|train_stage1|run_stage1_oof|ultralytics|stage1_oof|YOLO-CV"
    } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine

$drives = Get-CimInstance Win32_LogicalDisk |
    Select-Object DeviceID, VolumeName, DriveType,
        @{n="SizeGB";e={[math]::Round($_.Size / 1GB, 2)}},
        @{n="FreeGB";e={[math]::Round($_.FreeSpace / 1GB, 2)}}

$psDrives = Get-PSDrive -PSProvider FileSystem |
    Select-Object Name, Root, DisplayRoot,
        @{n="FreeGB";e={if ($_.Free -ne $null) {[math]::Round($_.Free / 1GB, 2)} else {$null}}},
        @{n="UsedGB";e={if ($_.Used -ne $null) {[math]::Round($_.Used / 1GB, 2)} else {$null}}}

$runsRoot = Join-Path $RepoRoot "YOLOv11\runs\stage1_oof_10fold"
$datasetRoot = Join-Path $aiRoot "datasets\final_sewerml_dataset"
$node = "unknown"
if (Test-Path -LiteralPath (Join-Path $runsRoot "fold_00")) { $node = "18" }
if (Test-Path -LiteralPath (Join-Path $runsRoot "fold_04")) { $node = "13" }

[ordered]@{
    generated_at = (Get-Date).ToString("s")
    computer = $env:COMPUTERNAME
    user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    repo_root = $RepoRoot
    detected_node = $node
    drives = @($drives)
    ps_drives = @($psDrives)
    matching_processes = @($procRows)
    paths = @($paths | ForEach-Object { Get-LinkInfo -Path $_ })
    dataset_reparse_ancestors = @(Get-ReparseAncestors -Path $datasetRoot)
    runs_reparse_ancestors = @(Get-ReparseAncestors -Path $runsRoot)
    run_status = @(Get-RunStatus -RunsRoot $runsRoot)
} | ConvertTo-Json -Depth 8
