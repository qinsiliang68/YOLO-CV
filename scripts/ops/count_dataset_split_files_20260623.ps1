$Dataset = "C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset"
$splits = @(
    "train",
    "normal_train",
    "val_model",
    "normal_val_model",
    "val_cal",
    "normal_val_cal",
    "val_op",
    "normal_val_op",
    "test",
    "normal_test"
)
foreach ($split in $splits) {
    $dir = Join-Path $Dataset ("Det\images\" + $split)
    if (-not (Test-Path -LiteralPath $dir)) {
        Write-Output "$split|exists=False|files=-1"
        continue
    }
    $count = (Get-ChildItem -LiteralPath $dir -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output "$split|exists=True|files=$count"
}
