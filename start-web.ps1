param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8023,
    [string]$HostAddress = "0.0.0.0"
)

$receiptPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $receiptPython)) {
    throw "Project .venv is missing. Restore it before starting the server."
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $receiptPython -m App.web_server --host $HostAddress --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Receipt server exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
