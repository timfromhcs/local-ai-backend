// Measure the tools suite for both models and persist the result, so a restart cannot lose it.
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const out = "docs/tools-bench.json";
const runs = [];

for (const model of ["qwen3-4b", "duo"]) {
  const started = Date.now();
  try {
    const res = await fetch(base + "/v1/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, suite: "tools" }),
      signal: AbortSignal.timeout(5400_000),
    });
    const body = await res.json();
    runs.push({
      model,
      status: res.status,
      score: body.score ?? null,
      seconds: +((Date.now() - started) / 1000).toFixed(1),
      rows: (body.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds, error: r.error })),
      error: body.error ?? null,
    });
  } catch (e) {
    runs.push({ model, error: String(e).slice(0, 200), seconds: +((Date.now() - started) / 1000).toFixed(1) });
  }
  console.log(model, JSON.stringify(runs.at(-1)).slice(0, 300));
}

fs.writeFileSync(out, JSON.stringify({ at: new Date().toISOString(), suite: "tools", runs }, null, 2));
console.log("written", out);
