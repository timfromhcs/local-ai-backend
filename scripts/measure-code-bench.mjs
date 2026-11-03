// Compare lanes on the CODE suite: the only suite where the score comes from running the tests.
// 1) qwen3-4b alone (tool loop, no planning)  2) duo forced (planner + verifier), same tool loop
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const runs = [];

for (const [model, extra] of [["qwen3-4b", {}], ["duo", { mode: "duo" }]]) {
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
      rows: (body.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds,
                                            answer: (r.answer || "").slice(0, 400), error: r.error })),
      error: body.error ?? null,
    });
  } catch (e) {
    runs.push({ model, lane: extra.mode ?? "auto", error: String(e).slice(0, 300),
                seconds: +((Date.now() - started) / 1000).toFixed(1) });
  }
  console.log(JSON.stringify(runs.at(-1)).slice(0, 500));
}

fs.writeFileSync("docs/code-bench.json", JSON.stringify({ at: new Date().toISOString(), suite: "code", runs }, null, 2));
console.log("written docs/code-bench.json");
