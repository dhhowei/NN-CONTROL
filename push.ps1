param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GitArgs
)

# Thin wrapper: push through whatever proxy network.cfg describes (local URL or share link).
$net = Join-Path $PSScriptRoot "net.ps1"
& $net @(@("git", "push") + $GitArgs)
exit $LASTEXITCODE
