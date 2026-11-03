# Compare the bare 4B against the duo virtual model on the same tasks, and write the result to
# docs/duo-bench.json. Kept separate from the cycle so a slow run cannot block a commit.
$ErrorActionPreference = 'Stop'
$out = 'C:\LocalAIBACKEND\docs\duo-bench.json'
$tasks = @('routing', 'tool-call', 'math', 'json')
$result = [ordered]@{ at = (Get-Date).ToString('s'); tasks = $tasks; runs = @() }

foreach ($model in @('qwen3-4b', 'duo')) {
    $body = @{ model = $model; suite = 'fast'; tasks = $tasks; max_tokens = 200 } | ConvertTo-Json
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-RestMethod 'http://127.0.0.1:8090/v1/bench' -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5400
        $sw.Stop()
        $result.runs += [ordered]@{
            model = $model; score = $r.score; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            by_category = $r.by_category
            rows = $r.rows | ForEach-Object { [ordered]@{ id = $_.id; score = $_.score; seconds = $_.seconds; answer = $_.answer.Substring(0, [Math]::Min(160, $_.answer.Length)) } }
        }
    } catch {
        $sw.Stop()
        $result.runs += [ordered]@{ model = $model; error = $_.Exception.Message; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
    }
}

$result | ConvertTo-Json -Depth 6 | Set-Content $out -Encoding utf8
Get-Content $out
