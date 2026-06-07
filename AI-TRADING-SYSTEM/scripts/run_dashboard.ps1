param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Url = "http://localhost:$Port"

Set-Location $ProjectRoot

$Existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $Existing) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        $Python = "python"
    }

    Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "$Port", "--server.headless", "true") `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden

    Start-Sleep -Seconds 5
}

$ChromeCandidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
)

$Chrome = $ChromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($Chrome) {
    Start-Process -FilePath $Chrome -ArgumentList $Url
} else {
    Start-Process $Url
}
