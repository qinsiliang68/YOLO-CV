param(
    [string]$TexFile = "C:\GitHub\YOLO-CV\essay\docs\essay.tex",
    [string]$OutputPdf = ""
)

$xelatex = "C:\Users\28898\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
$python = "C:\GitHub\YOLO-CV\.venv\Scripts\python.exe"
$renderScript = "C:\GitHub\YOLO-CV\scripts\render_thesis_format.py"

if ([string]::IsNullOrWhiteSpace($OutputPdf)) {
    $desktopDir = Join-Path $env:USERPROFILE "Desktop"
    $pdfName = ([string]([char]0x521D)) + ([char]0x7A3F) + "3.29.pdf"
    $OutputPdf = Join-Path $desktopDir $pdfName
}

$texPath = (Resolve-Path $TexFile).Path
$texDir = Split-Path -Parent $texPath
$texName = Split-Path -Leaf $texPath
$buildDir = Join-Path $texDir "build"

& $python $renderScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path -LiteralPath $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $texDir
try {
    & $xelatex "-interaction=nonstopmode" "-halt-on-error" "-output-directory=build" $texName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $xelatex "-interaction=nonstopmode" "-halt-on-error" "-output-directory=build" $texName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

$buildPdf = Join-Path $buildDir "essay.pdf"
Copy-Item -LiteralPath $buildPdf -Destination $OutputPdf -Force
$badOutputPdf = Join-Path (Split-Path -Parent $OutputPdf) "鍒濈3.29.pdf"
if (Test-Path -LiteralPath $badOutputPdf) {
    Remove-Item -LiteralPath $badOutputPdf -Force
}
Write-Host "[done] thesis pdf -> $OutputPdf"
