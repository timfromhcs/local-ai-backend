// Variance check: the same suite, the same models, three more runs. A single run is not evidence.
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const runs = [];

for (const [model, extra] of [["qwen3-4b", {}], ["qwen3-4b", {}], ["duo", { mode: "duo" }]]) {
  const started = Date.now();
  try {
    const res = await fetch(base + "/v1/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, suite: "code", max_tokens: 600, ...extra }),
      signal: AbortSignal.timeout(5400_000),
    });
    const body = await res.json();
    runs.push({
      model, lane: extra.mode ?? (model === "duo" ? "auto" : "model"), status: res.status,
      score: body.score ?? null, seconds: +((Date.now() - started) / 1000).toFixed(1),
      rows: (body.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds })),
      error: body.error ?? null,
    });
  } catch (e) {
    runs.push({ model, lane: extra.mode ?? "auto", error: String(e).slice(0, 200),
                seconds: +((Date.now() - started) / 1000).toFixed(1) });
  }
  console.log(JSON.stringify(runs.at(-1)));
}

fs.writeFileSync("docs/code-bench-reruns.json", JSON.stringify({ at: new Date().toISOString(), suite: "code", runs }, null, 2));
console.log("written docs/code-bench-reruns.json");
