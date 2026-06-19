param(
    [string]$AiRoot = "C:\Users\ASUS\Desktop\ssh\AI",
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

function Get-DirectorySizeGb {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1GB, 2)
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
        $sizeGb = Get-DirectorySizeGb -Path $Source
        Write-Step "[dry-run] robocopy /MOVE $Label ($sizeGb GB): $Source -> $Destination"
        return
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $script:StorageRootFinal "logs") | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $safeLabel = $Label -replace "[^A-Za-z0-9_.-]", "_"
    $log = Join-Path $script:StorageRootFinal "logs\robocopy_${safeLabel}_${stamp}.log"
    $sizeGb = Get-DirectorySizeGb -Path $Source
    Write-Step "robocopy /MOVE $Label ($sizeGb GB): $Source -> $Destination"
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
        [Parameter(Mandatory=$true)][string]$Name
    )

    $linkPath = Join-Path $AiRoot $Name
    $targetPath = Join-Path $script:StorageRootFinal $Name
    New-Item -ItemType Directory -Force -Path $targetPath | Out-Null

    if (Test-Path -LiteralPath $linkPath) {
        if (Get-IsReparse -Path $linkPath) {
            $currentTarget = Get-ReparseTarget -Path $linkPath
            if ($currentTarget.TrimEnd("\") -ieq $targetPath.TrimEnd("\")) {
                Write-Step "$Name already junctioned to $targetPath"
                return
            }
            Write-Step "$Name is a junction to $currentTarget; replacing with $targetPath"
            Remove-JunctionOnly -Path $linkPath
        } else {
            Invoke-RobocopyMove -Source $linkPath -Destination $targetPath -Label $Name
            Remove-DirectoryAfterMove -Path $linkPath -AllowedRoot $AiRoot
        }
    }

    if ($DryRun) {
        Write-Step "[dry-run] create junction: $linkPath -> $targetPath"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $linkPath) | Out-Null
    cmd /c mklink /J "$linkPath" "$targetPath" | Out-Host
}

function Ensure-SpecificJunction {
    param(
        [Parameter(Mandatory=$true)][string]$LinkPath,
        [Parameter(Mandatory=$true)][string]$TargetPath
    )
    if (Test-Path -LiteralPath $LinkPath) {
        if (Get-IsReparse -Path $LinkPath) {
            $currentTarget = Get-ReparseTarget -Path $LinkPath
            if ($currentTarget.TrimEnd("\") -ieq $TargetPath.TrimEnd("\")) {
                Write-Step "specific junction already correct: $LinkPath -> $TargetPath"
                return
            }
            Remove-JunctionOnly -Path $LinkPath
        } else {
            throw "Refusing to replace real directory with junction: $LinkPath"
        }
    }
    if ($DryRun) {
        Write-Step "[dry-run] create specific junction: $LinkPath -> $TargetPath"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LinkPath) | Out-Null
    New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null
    cmd /c mklink /J "$LinkPath" "$TargetPath" | Out-Host
}

function Detect-StorageRoot {
    $repoRuns = Join-Path $AiRoot "repos\YOLO-CV\YOLOv11\runs"
    if (Test-Path -LiteralPath $repoRuns) {
        if (Get-IsReparse -Path $repoRuns) {
            $target = Get-ReparseTarget -Path $repoRuns
            $marker = "\runs\YOLOv11"
            if ($target.EndsWith($marker, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $target.Substring(0, $target.Length - $marker.Length)
            }
        }
    }
    if (Test-Path -LiteralPath "F:\ssh\AI") { return "F:\ssh\AI" }
    if (Test-Path -LiteralPath "D:\ssh\AI") { return "D:\ssh\AI" }
    throw "Could not detect StorageRoot. Pass -StorageRoot."
}

if ([string]::IsNullOrWhiteSpace($StorageRoot)) {
    $StorageRoot = Detect-StorageRoot
}
if ($StorageRoot.Substring(0, 2).ToUpperInvariant() -eq "C:") {
    throw "StorageRoot must not be on C: $StorageRoot"
}
$script:StorageRootFinal = $StorageRoot

Write-Step "ai_root=$AiRoot storage=$StorageRoot dry_run=$DryRun"
$repoRunsLink = Join-Path $AiRoot "repos\YOLO-CV\YOLOv11\runs"
$repoRunsTarget = Join-Path $StorageRoot "runs\YOLOv11"
foreach ($name in @("repos", "projects", "venvs", "wheelhouse", "tmp", "cache")) {
    Ensure-JunctionToStorage -Name $name
}
Ensure-SpecificJunction -LinkPath $repoRunsLink -TargetPath $repoRunsTarget

Write-Step "code/env relocation completed"
