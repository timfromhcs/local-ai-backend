# Planner session: run duo with the 27B planner, which only fits if the fast lane is stopped.
#
# Measured 2026-11-04 (docs/bench-log.md): the 27B planner does not load next to the 1.5B fast lane
# (the shared VRAM carve-out is too small), but with the fast lane stopped the pair loads and duo
# scores 0.667 on the hard suite instead of 0.333.
#
#   .\scripts\planner-session.ps1 -Start     stop the fast lane, start the gateway with planner=bonsai-27b
#   .\scripts\planner-session.ps1 -Stop      restore the fast lane and the default gateway
#   .\scripts\planner-session.ps1 -Status    show which servers are running
#
# This is a deliberate trade: no fast lane and no mode routing while the session runs, so send
# localai_mode explicitly (e.g. "duo") instead of relying on the controller.
param(
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$jevGuard = Join-Path $env:USERPROFILE ".config\opencode\scripts\jev-guard.ps1"
$gatewayPort = 8090

function Stop-Gateway {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*localai_gateway*$gatewayPort*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Seconds 2
}

function Stop-FastLane {
    Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
        Where-Object { $_.CommandLine -like "*--port 8080*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Seconds 2
}

function Stop-Backends {
    Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
        Where-Object { $_.CommandLine -like "*LocalAIBACKEND*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Seconds 2
}

function Start-Gateway([hashtable]$env) {
    $saved = @{}
    foreach ($k in $env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $env[$k]) }
    try {
        Start-Process -FilePath $python -ArgumentList "-m", "localai_gateway", "--port", "$gatewayPort" `
            -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $root "rsi\logs\gateway.log") `
            -RedirectStandardError (Join-Path $root "rsi\logs\gateway.err") -WindowStyle Hidden
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
    Start-Sleep -Seconds 6
    (Invoke-RestMethod "http://127.0.0.1:$gatewayPort/healthz" -TimeoutSec 10).status
}

if ($Start) {
    Write-Host "stopping fast lane and backends (they cannot share VRAM with the 27B planner)"
    Stop-Gateway
    Stop-FastLane
    Stop-Backends
    $status = Start-Gateway @{ LOCALAI_DUO_PLANNER = "bonsai-27b"; LOCALAI_DUO_VERIFIER = "bonsai-27b"; LOCALAI_BENCH_EXEC = "1" }
    Write-Host "gateway $status on :$gatewayPort with planner/verifier = bonsai-27b"
    Write-Host "send localai_mode explicitly - the fast lane is down, so the mode controller has no Jey server"
    exit 0
}

if ($Stop) {
    Write-Host "restoring fast lane and default gateway"
    Stop-Gateway
    Stop-Backends
    if (Test-Path $jevGuard) { & $jevGuard -Restart | Select-Object -Last 1 }
    $status = Start-Gateway @{ LOCALAI_BENCH_EXEC = "1" }
    Write-Host "gateway $status on :$gatewayPort with the default role assignment"
    exit 0
}

# -Status (default)
$servers = Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" | ForEach-Object {
    $id = $_.ProcessId
    $mb = [int]((Get-Process -Id $id -ErrorAction SilentlyContinue).WorkingSet64 / 1MB)
    $port = [regex]::Match($_.CommandLine, "--port (\d+)").Groups[1].Value
    "  llama-server pid=$id port=$port ${mb}MB"
}
$gateway = try { (Invoke-RestMethod "http://127.0.0.1:$gatewayPort/healthz" -TimeoutSec 5).status } catch { "down" }
Write-Host "gateway :$gatewayPort $gateway"
$servers
