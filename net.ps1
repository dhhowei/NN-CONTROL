<#
Run any network command through the proxy described in network.cfg.

network.cfg may contain either:
  * a local proxy URL:   socks5://127.0.0.1:10808 | http://... | https://...
  * a share link:        vmess://... | vless://... | trojan://... | ss://...
    (for share links this script generates an Xray config and starts a local
     SOCKS inbound automatically, then stops it when the command finishes)

Usage:
  .\net.ps1 git push
  .\net.ps1 git ls-remote origin HEAD
  .\net.ps1 gh api /user
  .\net.ps1 curl https://api.github.com
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cfgPath = Join-Path $root "network.cfg"
$generator = Join-Path $root "tools\make_xray_config.py"

function Get-Entry([string]$path) {
    foreach ($line in Get-Content -LiteralPath $path) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith("#")) { continue }
        $iEq = $t.IndexOf("="); $iSch = $t.IndexOf("://")
        if ($iEq -ge 0 -and ($iSch -lt 0 -or $iEq -lt $iSch)) { $t = $t.Substring($iEq + 1).Trim() }
        return $t
    }
    return $null
}

function Get-FreePort([int]$start) {
    for ($p = $start; $p -lt ($start + 200); $p++) {
        try {
            $l = [System.Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $p)
            $l.Start(); $l.Stop(); return $p
        } catch { }
    }
    throw "no free port found near $start"
}

function Wait-Port([int]$port, [int]$timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $c = [System.Net.Sockets.TcpClient]::new()
            $c.Connect("127.0.0.1", $port); $c.Close(); return $true
        } catch { Start-Sleep -Milliseconds 150 }
    }
    return $false
}

function Find-Xray {
    if ($env:XRAY_EXE -and (Test-Path -LiteralPath $env:XRAY_EXE)) { return $env:XRAY_EXE }
    $cmd = Get-Command xray.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $roots = @("$env:USERPROFILE\Downloads", "$env:USERPROFILE\Desktop", "$env:USERPROFILE")
    foreach ($r in $roots) {
        if (-not (Test-Path -LiteralPath $r)) { continue }
        $f = Get-ChildItem -LiteralPath $r -Recurse -Depth 5 -Filter xray.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($f) { return $f.FullName }
    }
    return $null
}

$entry = $null
if (Test-Path -LiteralPath $cfgPath) { $entry = Get-Entry $cfgPath }

$proxy = $null
$coreProc = $null
$coreCfg = $null
$saved = @{}

if ($entry) {
    $scheme = ($entry -split "://")[0].ToLower()
    if ($scheme -in @("socks5", "socks5h", "http", "https")) {
        $proxy = $entry
    } elseif ($scheme -in @("vmess", "vless", "trojan", "ss")) {
        $xray = Find-Xray
        if (-not $xray) { throw "xray.exe not found. Set XRAY_EXE or install V2RayN/Xray." }
        $port = Get-FreePort 10888
        $coreCfg = Join-Path $env:TEMP ("netxray_" + [guid]::NewGuid().ToString("N") + ".json")
        & python $generator --config $cfgPath --out $coreCfg --socks-port $port | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "failed to generate xray config" }
        $coreProc = Start-Process -FilePath $xray -ArgumentList @("-c", $coreCfg) -PassThru -WindowStyle Hidden
        if (-not (Wait-Port $port 15)) {
            Stop-Process -Id $coreProc.Id -Force -ErrorAction SilentlyContinue
            throw "xray did not open 127.0.0.1:$port"
        }
        $proxy = "socks5://127.0.0.1:$port"
        Write-Host "[net] started xray (pid $($coreProc.Id)) for $scheme link, socks=$proxy"
    }
}

if ($proxy) {
    $saved.HTTP_PROXY = $env:HTTP_PROXY; $saved.HTTPS_PROXY = $env:HTTPS_PROXY; $saved.ALL_PROXY = $env:ALL_PROXY
    $saved.http_proxy = $env:http_proxy; $saved.https_proxy = $env:https_proxy; $saved.all_proxy = $env:all_proxy
    $env:HTTP_PROXY = $proxy; $env:HTTPS_PROXY = $proxy; $env:ALL_PROXY = $proxy
    $env:http_proxy = $proxy; $env:https_proxy = $proxy; $env:all_proxy = $proxy
    Write-Host "[net] using proxy: $proxy"
} else {
    Write-Host "[net] no proxy found - running directly."
}

try {
    if ($Command.Count -eq 0) { throw "no command given" }
    & $Command[0] @($Command[1..($Command.Count - 1)])
    $code = $LASTEXITCODE
} finally {
    foreach ($k in $saved.Keys) { Set-Item -Path "env:$k" -Value $saved[$k] }
    if ($coreProc) { Stop-Process -Id $coreProc.Id -Force -ErrorAction SilentlyContinue }
    if ($coreCfg -and (Test-Path -LiteralPath $coreCfg)) { Remove-Item -LiteralPath $coreCfg -Force -ErrorAction SilentlyContinue }
}
exit $code
