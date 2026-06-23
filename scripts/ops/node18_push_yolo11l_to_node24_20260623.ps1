$ErrorActionPreference = "Stop"

$Source = "D:\ssh\AI\repos\YOLO-CV\yolo11l-cls.pt"
if (-not (Test-Path -LiteralPath $Source)) {
    $Source = "C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV\yolo11l-cls.pt"
}
if (-not (Test-Path -LiteralPath $Source)) {
    throw "Missing source weight yolo11l-cls.pt"
}

$Target = "ASUS@192.168.100.24:C:/Users/ASUS/Desktop/ssh/AI/repos/YOLO-CV/yolo11l-cls.pt"
$Key = "C:\Users\ASUS\.ssh\node_to_node_ed25519"

scp -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL $Source $Target
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Source
Write-Output ("WEIGHT_PUSH_OK source={0} sha256={1}" -f $Source, $hash.Hash.ToLowerInvariant())
