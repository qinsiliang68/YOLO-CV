param(
    [string]$BaseDir = "C:\GitHub\YOLO-CV\data\sewerml",
    [switch]$Extract,
    [ValidateSet("HardLink", "Copy")]
    [string]$Mode = "HardLink",
    [string[]]$Splits = @("Train", "Val", "Test")
)

$ErrorActionPreference = "Stop"

$DefectLabels = @(
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF",
    "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK"
)
$AllLabels = @("Normal") + $DefectLabels

function Ensure-Structure {
    param([string]$Root)

    $archives = Join-Path $Root "archives"
    $images = Join-Path $Root "images_all"
    $annotations = Join-Path $Root "annotations"
    $byClass = Join-Path $Root "by_class"

    foreach ($dir in @($archives, $images, $annotations, $byClass)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    foreach ($split in @("Train", "Val", "Test")) {
        foreach ($label in $AllLabels) {
            New-Item -ItemType Directory -Force -Path (Join-Path $byClass (Join-Path $split $label)) | Out-Null
        }
    }

    return @{
        Archives = $archives
        Images = $images
        Annotations = $annotations
        ByClass = $byClass
    }
}

function Resolve-CsvPath {
    param(
        [string]$AnnotationsDir,
        [string]$Split
    )

    $candidates = @(
        (Join-Path $AnnotationsDir "$Split`13.csv"),
        (Join-Path $AnnotationsDir "SewerML_$Split.csv")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

function New-LinkOrCopy {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$PlacementMode
    )

    if (Test-Path -LiteralPath $Destination) {
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null

    if ($PlacementMode -eq "HardLink") {
        try {
            New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
            return
        }
        catch {
        }
    }

    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Extract-Archives {
    param(
        [string]$ArchivesDir,
        [string]$ImagesDir
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archives = Get-ChildItem -LiteralPath $ArchivesDir -Filter *.zip | Sort-Object Name
    foreach ($archive in $archives) {
        Write-Host "[extract] $($archive.Name)"
        $zip = [System.IO.Compression.ZipFile]::OpenRead($archive.FullName)
        try {
            $entries = $zip.Entries | Where-Object {
                $ext = [System.IO.Path]::GetExtension($_.FullName).ToLowerInvariant()
                $ext -in @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
            }

            $count = 0
            foreach ($entry in $entries) {
                $count++
                $target = Join-Path $ImagesDir ([System.IO.Path]::GetFileName($entry.FullName))
                if (-not (Test-Path -LiteralPath $target)) {
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target)
                }

                if (($count % 5000) -eq 0) {
                    Write-Host "  extracted $count"
                }
            }
            Write-Host "  extracted total $count"
        }
        finally {
            $zip.Dispose()
        }
    }
}

function Organize-Split {
    param(
        [string]$CsvPath,
        [string]$ImagesDir,
        [string]$SplitDir,
        [string]$PlacementMode
    )

    Write-Host "[organize] $(Split-Path $CsvPath -Leaf)"
    $rows = Import-Csv -LiteralPath $CsvPath
    $created = 0
    $missing = 0
    $normal = 0

    foreach ($row in $rows) {
        $filename = Split-Path -Leaf $row.Filename
        $source = Join-Path $ImagesDir $filename
        if (-not (Test-Path -LiteralPath $source)) {
            $missing++
            continue
        }

        $positiveLabels = @()
        foreach ($label in $DefectLabels) {
            if ($null -ne $row.$label -and [string]$row.$label -eq "1") {
                $positiveLabels += $label
            }
        }

        if (($null -ne $row.Defect -and [string]$row.Defect -eq "0") -or $positiveLabels.Count -eq 0) {
            $target = Join-Path $SplitDir (Join-Path "Normal" $filename)
            New-LinkOrCopy -Source $source -Destination $target -PlacementMode $PlacementMode
            $created++
            $normal++
        }

        foreach ($label in $positiveLabels) {
            $target = Join-Path $SplitDir (Join-Path $label $filename)
            New-LinkOrCopy -Source $source -Destination $target -PlacementMode $PlacementMode
            $created++
        }
    }

    Write-Host "  created entries: $created"
    Write-Host "  normal entries: $normal"
    if ($missing -gt 0) {
        Write-Host "  missing source images: $missing"
    }
}


$paths = Ensure-Structure -Root $BaseDir

if ($Extract) {
    Extract-Archives -ArchivesDir $paths.Archives -ImagesDir $paths.Images
}

$foundCsv = $false
foreach ($split in $Splits) {
    $csvPath = Resolve-CsvPath -AnnotationsDir $paths.Annotations -Split $split
    if (-not $csvPath) {
        Write-Host "[skip] missing annotation file for split: $split"
        continue
    }

    $foundCsv = $true
    Organize-Split -CsvPath $csvPath -ImagesDir $paths.Images -SplitDir (Join-Path $paths.ByClass $split) -PlacementMode $Mode
}

if (-not $foundCsv) {
    Write-Host "[done] no official CSV found yet."
    Write-Host "Place Train13.csv / Val13.csv / Test13.csv into: $($paths.Annotations)"
}
