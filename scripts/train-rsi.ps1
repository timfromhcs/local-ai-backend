# Plan or run LoRA training on captured RSI pairs in the training venv.
#   .\scripts\train-rsi.ps1                # plan only (seconds)
#   .\scripts\train-rsi.ps1 -Run           # real CPU run (hours, checkpoints)
#   .\scripts\train-rsi.ps1 -Run -Epochs 2 -CheckpointEvery 10
[CmdletBinding()]
param(
    [switch]$Run,
    [int]$Epochs = 1,
    [int]$Rank = 8,
    [int]$CheckpointEvery = 20,
    [string]$Pairs = "rsi/captures/*.pairs.jsonl",
    [string]$Model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv-train\Scripts\python.exe'
if (-not (Test-Path $venv)) {
    throw "training venv missing: $venv. Create it with: python -m venv .venv-train && .venv-train\Scripts\pip install torch transformers datasets peft accelerate"
}

$log = Join-Path $root 'rsi\runs\train.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# the training venv is intentionally separate from .venv, so it does not have the package
# installed; put src on the path instead of pip-installing twice
$env:PYTHONPATH = Join-Path $root 'src'
$env:HF_HOME = Join-Path $root 'cache\hf'

$args = @('-m', 'rsi_lab.train', '--pairs', (Join-Path $root $Pairs), '--epochs', $Epochs,
          '--rank', $Rank, '--checkpoint-every', $CheckpointEvery, '--model', $Model)
if ($Run) { $args += '--run' }

Write-Host "python $($args -join ' ')"
& $venv @args 2>&1 | Tee-Object -Append -FilePath $log
exit $LASTEXITCODE
