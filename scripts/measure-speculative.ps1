# Measure speculative decoding for a TARGET + DRAFT pair with the same tokenizer family.
#   .\measure-speculative.ps1 -Target <gguf> -Draft <gguf> -Label tag [-DraftMax 3]
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(Mandatory = $true)][string]$Draft,
    [Parameter(Mandatory = $true)][string]$Label,
    [int]$DraftMax = 3,
    [int]$Port = 18121
)

$ErrorActionPreference = 'Stop'
$root = 'C:\LocalAIBACKEND'
$server = Join-Path $root 'bin\llama\llama-server.exe'
$prompt = 'List the first eight prime numbers and explain in one sentence why 1 is not prime.'

function Wait-Healthy([int]$p, [int]$timeoutSec = 180) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$p/health" -TimeoutSec 2).status -eq 'ok') { return $true } } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Measure-Run([string]$tag, [string[]]$extra) {
    $args = @('-m', $Target, '--host', '127.0.0.1', '--port', "$Port", '-c', '2048', '-ngl', '99',
              '-fa', 'on', '-t', '8', '-ub', '256', '-np', '1', '--no-warmup') + $extra
    $log = Join-Path $root "rsi\logs\$Label-$tag.log"
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    $proc = Start-Process -FilePath $server -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
    try {
        if (-not (Wait-Healthy $Port)) { return @{ label = $tag; error = 'server did not become healthy'; log = $log } }
        $body = @{ prompt = $prompt; n_predict = 8; temperature = 0; cache_prompt = $true } | ConvertTo-Json
        Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300 | Out-Null
        $body = @{ prompt = "$prompt Run $tag."; n_predict = 64; temperature = 0; cache_prompt = $false } | ConvertTo-Json
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
        $specDisabled = Select-String -Path "$log.err" -Pattern 'speculative decoding .* disabled' -Quiet
        return @{
            label = $tag
            decode_tps = [math]::Round($r.timings.predicted_per_second, 2)
            prompt_tps = [math]::Round($r.timings.prompt_per_second, 2)
            tokens = $r.timings.predicted_n
            prompt_tokens = $r.timings.prompt_n
            spec_disabled = [bool]$specDisabled
            log = $log
        }
    } finally {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
        Start-Sleep -Seconds 3
    }
}

$results = @()
$results += Measure-Run 'baseline' @()
$results += Measure-Run 'spec' @('-md', $Draft, '--spec-type', 'draft-simple', '--spec-draft-n-max', "$DraftMax", '--spec-draft-n-min', '1')
$out = Join-Path $root "rsi\logs\spec-$Label.json"
$results | ConvertTo-Json -Depth 4 | Set-Content $out -Encoding utf8
$results | Format-Table label, decode_tps, prompt_tps, tokens, spec_disabled -AutoSize
