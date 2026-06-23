$paths = @(
    "C:\Users\ASUS\Desktop\ssh\AI\repos",
    "C:\Users\ASUS\Desktop\ssh\AI\runs",
    "C:\Users\ASUS\Desktop\ssh\AI\artifacts",
    "C:\Users\ASUS\Desktop\ssh\AI\logs",
    "C:\Users\ASUS\Desktop\ssh\AI\workdirs",
    "C:\Users\ASUS\Desktop\ssh\AI\datasets"
)
foreach ($path in $paths) {
    $item = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if (-not $item) {
        Write-Output "$path|exists=False|reparse=False|target="
        continue
    }
    $isReparse = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
    $target = ""
    if ($isReparse) {
        $target = ($item.Target -join ";")
    }
    Write-Output "$path|exists=True|reparse=$isReparse|target=$target"
}
