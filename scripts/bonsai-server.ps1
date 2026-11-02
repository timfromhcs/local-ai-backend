# Start local Bonsai-27B for OpenCode (implementer). Writes its own PID to .superpowers\probe\bonsai.pid.
# Stop with: scripts\bonsai-server.ps1 -Stop   (kills only the PID recorded here, never by image name).
param([switch]$Stop, [int]$Port = 18090, [int]$Ctx = 4096)
$root = Split-Path $PSScriptRoot -Parent
$pidFile = Join-Path $root ".superpowers\probe\bonsai.pid"
New-Item -ItemType Directory -Force (Split-Path $pidFile) | Out-Null
if ($Stop) {
    if (Test-Path $pidFile) {
        $p = [int](Get-Content $pidFile)
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile
        "stopped $p"
    } else { "no pid file" }
    return
}
$exe = Join-Path $root "bin\llama\llama-server.exe"
$model = Join-Path $root "Bonsai-2-27B-Philadelphia-Class-MIXED-PTQ1-Q4_0.gguf"
$log = Join-Path $root ".superpowers\probe\bonsai-opencode.log"
$args = @("--model", $model, "--host", "127.0.0.1", "--port", $Port, "-c", $Ctx, "-ngl", "99", "-fa", "on",
          "-ctk", "q8_0", "-ctv", "q8_0", "--no-warmup", "-np", "1", "-t", "8", "--reasoning-budget", "0")
$proc = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardError $log
$proc.Id | Set-Content $pidFile
"started pid $($proc.Id) on port $Port"
