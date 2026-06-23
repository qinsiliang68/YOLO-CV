$ErrorActionPreference = "Continue"

$Repo = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV"
$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$VenvPython = "C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe"
$Uv = "C:\Users\ASUS\.local\bin\uv.exe"

function Info {
    param([string]$Name, [object]$Value)
    Write-Output ("{0}={1}" -f $Name, $Value)
}

function Test-Reparse {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $item) { return $false }
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

Info hostname $env:COMPUTERNAME
Info whoami ([Security.Principal.WindowsIdentity]::GetCurrent().Name)

Info repo_exists (Test-Path -LiteralPath $Repo)
Info repo_git (Test-Path -LiteralPath (Join-Path $Repo ".git"))
if (Test-Path -LiteralPath (Join-Path $Repo ".git")) {
    Info repo_branch (& git -C $Repo rev-parse --abbrev-ref HEAD 2>$null)
    Info repo_commit (& git -C $Repo rev-parse --short HEAD 2>$null)
    Info repo_status ((& git -C $Repo status --porcelain 2>$null | Select-Object -First 12) -join "|")
}

Info dataset_exists (Test-Path -LiteralPath $Dataset)
if (Test-Path -LiteralPath $Dataset) {
    $item = Get-Item -LiteralPath $Dataset
    Info dataset_reparse (Test-Reparse $Dataset)
    Info dataset_fullname $item.FullName
    Info dataset_drive ([IO.Path]::GetPathRoot($item.FullName))
}

$manifestRoot = Join-Path $Dataset "manifests"
foreach ($name in @(
    "train_manifest.csv",
    "normal_train_manifest.csv",
    "val_cal_manifest.csv",
    "normal_val_cal_manifest.csv",
    "val_op_manifest.csv",
    "normal_val_op_manifest.csv",
    "test_manifest.csv",
    "normal_test_manifest.csv",
    "val_model_manifest.csv",
    "normal_val_model_manifest.csv"
)) {
    Info ("manifest_" + $name) (Test-Path -LiteralPath (Join-Path $manifestRoot $name))
}

Info venv_exists (Test-Path -LiteralPath $VenvPython)
Info uv_exists (Test-Path -LiteralPath $Uv)
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
Info git_path $(if ($gitCommand) { $gitCommand.Source } else { "" })
$nvidiaCommand = Get-Command nvidia-smi -ErrorAction SilentlyContinue
Info nvidia_smi_path $(if ($nvidiaCommand) { $nvidiaCommand.Source } else { "" })

Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    Info ("disk_" + $_.DeviceID.TrimEnd(":")) ("freeGB={0};sizeGB={1}" -f [math]::Round($_.FreeSpace / 1GB, 2), [math]::Round($_.Size / 1GB, 2))
}

Info gpu ((& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null) -join ";")
Info compute_procs ((& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null) -join ";")

$trainProcs = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "train_stage1|run_stage1|hn_rn|ultralytics|YOLO_CalOp|evaluate_stage1|phase1" }
)
Info train_proc_count $trainProcs.Count
$trainProcs | Select-Object -First 5 | ForEach-Object {
    Info train_proc ("{0}:{1}" -f $_.ProcessId, ($_.CommandLine -replace "\s+", " "))
}

if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython -c "import json, torch, torchvision, torchaudio, ultralytics; print('ENV_JSON=' + json.dumps({'python_ok': True, 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'torchaudio': torchaudio.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda': torch.version.cuda, 'gpu_count': torch.cuda.device_count(), 'gpu0': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'ultralytics': ultralytics.__version__}, ensure_ascii=False))"
}
