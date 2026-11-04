# Planner session: run duo with the 27B planner, which only fits if the fast lane is stopped.
#
# Measured 2026-11-04 (docs/bench-log.md): the 27B planner does not load next to the 1.5B fast lane
# (the shared VRAM carve-out is too small). With the fast lane stopped the pair loads, and duo runs
# with a genuinely stronger planner - but the log also records that a fair scorer shows no accuracy
# gain on the current hard suite, so treat this as a deliberate deep-work mode, not an upgrade.
#
#   .\scripts\planner-session.ps1 -Start     stop the fast lane, start the gateway with planner=bonsai-27b
#   .\scripts\planner-session.ps1 -Stop      restore the fast lane and the default gateway
#   .\scripts\planner-session.ps1 -Status    show which servers are running
#
# While the session runs there is no fast lane and no automatic mode routing, so send
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
$logDir = Join-Path $root "rsi\logs"

function Stop-Matching([string]$filter, [string]$pattern) {
    Get-CimInstance Win32_Process -Filter $filter |
        Where-Object { $_.CommandLine -like $pattern } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# Detach through cmd `start /b`: Start-Process with -RedirectStandardOutput keeps the caller attached
# to the child's streams, which made this script hang until the harness timed out (observed twice).
function Start-GatewayDetached([hashtable]$overrides) {
    $prefix = ($overrides.GetEnumerator() | ForEach-Object { "set `"$($_.Key)=$($_.Value)`" && " }) -join ""
    $cmd = "$prefix `"$python`" -m localai_gateway --port $gatewayPort > `"$logDir\gateway.log`" 2> `"$logDir\gateway.err`""
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "start", "/b", "cmd", "/c", $cmd -WorkingDirectory $root -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try { return (Invoke-RestMethod "http://127.0.0.1:$gatewayPort/healthz" -TimeoutSec 3).status } catch { }
    }
    return "not answering on :$gatewayPort"
}

if ($Start) {
    Write-Host "stopping fast lane and backends (they cannot share VRAM with the 27B planner)"
    Stop-Matching "name='python.exe'" "*localai_gateway*$gatewayPort*"
    Stop-Matching "name='llama-server.exe'" "*--port 8080*"
    Stop-Matching "name='llama-server.exe'" "*LocalAIBACKEND*"
    $health = Start-GatewayDetached @{ LOCALAI_DUO_PLANNER = "bonsai-27b"; LOCALAI_DUO_VERIFIER = "bonsai-27b"; LOCALAI_BENCH_EXEC = "1" }
    Write-Host "gateway $health on :$gatewayPort with planner/verifier = bonsai-27b"
    Write-Host "send localai_mode explicitly - the fast lane is down, so the mode controller has no Jev server"
    exit 0
}

if ($Stop) {
    Write-Host "restoring fast lane and default gateway"
    Stop-Matching "name='python.exe'" "*localai_gateway*$gatewayPort*"
    Stop-Matching "name='llama-server.exe'" "*LocalAIBACKEND*"
    if (Test-Path $jevGuard) { & $jevGuard -Restart | Select-Object -Last 1 }
    $health = Start-GatewayDetached @{ LOCALAI_BENCH_EXEC = "1" }
    Write-Host "gateway $health on :$gatewayPort with the default role assignment"
    exit 0
}

# -Status (default)
$health = try { (Invoke-RestMethod "http://127.0.0.1:$gatewayPort/healthz" -TimeoutSec 5).status } catch { "down" }
Write-Host "gateway :$gatewayPort $health"
Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" | ForEach-Object {
    $id = $_.ProcessId
    $mb = [int]((Get-Process -Id $id -ErrorAction SilentlyContinue).WorkingSet64 / 1MB)
    $port = [regex]::Match($_.CommandLine, "--port (\d+)").Groups[1].Value
    Write-Host "  llama-server pid=$id port=$port ${mb}MB"
}
