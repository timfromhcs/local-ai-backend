# Build a single-file localai-gateway.exe (no model weights, no llama/sd runtimes).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build-exe.ps1
[CmdletBinding()]
param([switch]$SkipTests)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$dist = Join-Path $root 'dist'
$work = Join-Path $root 'build'

if (-not (Test-Path $py)) { throw "virtualenv not found: $py" }

if (-not $SkipTests) {
    Write-Host '== pytest (not real) =='
    & $py -m pytest -m 'not real' -q
    if ($LASTEXITCODE -ne 0) { throw 'tests failed - refusing to build' }
}

Write-Host '== pyinstaller =='
& $py -m PyInstaller --noconfirm --clean --distpath $dist --workpath $work (Join-Path $root 'packaging\localai-gateway.spec')
if ($LASTEXITCODE -ne 0) { throw 'pyinstaller failed' }

$exe = Join-Path $dist 'localai-gateway.exe'
if (-not (Test-Path $exe)) { throw "exe missing: $exe" }
$sizeMb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "== built $exe ($sizeMb MB) =="

Write-Host '== smoke test: --version and --doctor =='
& $exe --version
& $exe --doctor | Select-Object -First 20
Write-Host '(doctor exit code is expected to be 1/0 depending on whether models are present)'

Write-Host '== smoke test: start server on a free port, call /healthz =='
$port = 18090
$proc = Start-Process -FilePath $exe -ArgumentList @('--port', "$port") -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/healthz" -TimeoutSec 2
        if ($r.status -eq 'ok') { $ok = $true; break }
    } catch { }
}
try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
if (-not $ok) { throw 'exe smoke test failed: /healthz did not answer' }
Write-Host '== exe smoke test PASSED =='
