# Deploy the static demo with a fresh service-worker cache version, then verify production.
#   .\scripts\deploy-web.ps1            # stamp, deploy, verify
#   .\scripts\deploy-web.ps1 -NoVerify  # stamp and deploy only
[CmdletBinding()]
param([switch]$NoVerify)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$web = Join-Path $root 'web'

Push-Location $root
try {
    # the gate must be green before anything goes out
    node scripts/check-web.mjs
    if ($LASTEXITCODE -ne 0) { throw 'web gate failed - not deploying' }

    node scripts/stamp-sw.mjs
    if ($LASTEXITCODE -ne 0) { throw 'could not stamp the service worker' }

    Push-Location $web
    try {
        vercel deploy --prod --yes
        if ($LASTEXITCODE -ne 0) { throw 'vercel deploy failed' }
    } finally { Pop-Location }

    if (-not $NoVerify) {
        node scripts/verify-pwa-production.mjs
        if ($LASTEXITCODE -ne 0) { throw 'production verification failed' }
    }
} finally { Pop-Location }
