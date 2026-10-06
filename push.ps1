param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GitArgs
)

$ErrorActionPreference = "Stop"
$cfg = Join-Path $PSScriptRoot "network.cfg"
$proxy = $null

if (Test-Path -LiteralPath $cfg) {
    foreach ($line in Get-Content -LiteralPath $cfg) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith("#")) { continue }
        $val = ($t -split "=", 2)[-1].Trim()
        if ($val -match "^(socks5h?|https?)://") { $proxy = $val }
    }
}

if ($proxy) {
    Write-Host "Using proxy from network.cfg: $proxy"
    git -c "http.proxy=$proxy" push @GitArgs
} else {
    Write-Host "No proxy in network.cfg - pushing directly."
    git push @GitArgs
}
