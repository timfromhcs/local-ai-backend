// Compare lanes on the HARD suite: tasks the 4B demonstrably fails.
// 1) qwen3-4b alone  2) duo forced to plan/execute/verify
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const runs = [];

for (const [model, extra] of [["qwen3-4b", {}], ["duo", { mode: "duo" }]]) {
  const started = Date.now();
  try {
    const res = await fetch(base + "/v1/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, suite: "hard", max_tokens: 240, ...extra }),
      signal: AbortSignal.timeout(5400000),
    });
    const body = await res.json();
    runs.push({
      model, mode: extra.mode ?? "auto", status: res.status,
      score: body.score ?? null, seconds: +((Date.now() - started) / 1000).toFixed(1),
      rows: (body.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds, answer: (r.answer || "").slice(0, 200), error: r.error })),
      error: body.error ?? null,
    });
  } catch (e) {
    runs.push({ model, mode: extra.mode ?? "auto", error: String(e).slice(0, 200), seconds: +((Date.now() - started) / 1000).toFixed(1) });
  }
  console.log(JSON.stringify(runs.at(-1)).slice(0, 400));
}

fs.writeFileSync("docs/hard-bench.json", JSON.stringify({ at: new Date().toISOString(), suite: "hard", runs }, null, 2));
console.log("written docs/hard-bench.json");
