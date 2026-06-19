param(
    [string]$RepoRoot = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV",
    [string]$TrainPython = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe",
    [string]$Uv = "C:\Users\ASUS\.local\bin\uv.exe"
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

Write-Host "repo=$RepoRoot"
Write-Host "python=$TrainPython"
Write-Host "uv=$Uv"

& $TrainPython -m py_compile `
    "scripts\continue_stage1_oof_node_20260619.py" `
    "scripts\run_stage1_oof_folds_20260617.py" `
    "scripts\train_stage1_cls_sweep.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $TrainPython -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Uv run --no-sync python "scripts\continue_stage1_oof_node_20260619.py" --print-only
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
