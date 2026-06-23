$ErrorActionPreference = "Stop"

$PubPath = "C:\Users\ASUS\Desktop\ssh\AI\tmp\node18_node_to_node_ed25519.pub.txt"
if (-not (Test-Path -LiteralPath $PubPath)) {
    throw "Missing public key file: $PubPath"
}
$PubKey = (Get-Content -Raw -LiteralPath $PubPath).Trim()
if (-not $PubKey.StartsWith("ssh-ed25519 ")) {
    throw "Unexpected public key content: $PubKey"
}

$UserSsh = Join-Path $env:USERPROFILE ".ssh"
$UserAuth = Join-Path $UserSsh "authorized_keys"
$AdminAuth = "C:\ProgramData\ssh\administrators_authorized_keys"
New-Item -ItemType Directory -Force -Path $UserSsh, (Split-Path $AdminAuth -Parent) | Out-Null

foreach ($path in @($UserAuth, $AdminAuth)) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType File -Force -Path $path | Out-Null
    }
    $existing = @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)
    if ($existing -notcontains $PubKey) {
        Add-Content -LiteralPath $path -Encoding ascii -Value $PubKey
        Write-Output "ADDED $path"
    } else {
        Write-Output "EXISTS $path"
    }
}
