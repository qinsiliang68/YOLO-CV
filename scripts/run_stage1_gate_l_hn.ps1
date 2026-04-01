param(
    [string]$ScoreDevice = "cpu",
    [int]$TopK = 22,
    [int]$ScoreBatch = 2
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$weights = "C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/runs/cls_gate_source/yolo11l_gate2_train7200/weights/best.pt"
$dataRoot = "C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/datasets/sewerml_gate2_train7200"
$outputDir = "C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/research/materials/stage1_hn/yolo11l_gate2_train7200"
$hnDataset = "C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/datasets/stage1_gate_hn_backflow/yolo11l_gate2_hn02"
$config = ".\\YOLOv11\\configs\\runtime\\stage1_gate_l_hn.json"

Write-Host "[stage1] score train-side normals for yolo11l-cls"
uv run python .\scripts\stage1_score_train_normals.py `
  --weights $weights `
  --data-root $dataRoot `
  --output-dir $outputDir `
  --device $ScoreDevice `
  --imgsz 640 `
  --batch $ScoreBatch `
  --top-k $TopK
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[stage1] build HN 2% dataset for yolo11l-cls"
uv run python .\scripts\stage1_build_hn_dataset.py `
  --source-dataset $dataRoot `
  --scores-csv "$outputDir/top_false_positive_normals.csv" `
  --output-dataset $hnDataset `
  --top-k $TopK `
  --repeat 1 `
  --link-mode hardlink
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[stage1] train yolo11l-cls + HN 2%"
uv run python .\scripts\stage1_gate_train.py --config $config
exit $LASTEXITCODE
