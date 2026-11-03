// Measure duo with the plan/execute/verify loop FORCED, on the tools suite.
// Without localai_mode the controller picks the fast lane and the loop never runs.
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const outFile = "docs/duo-forced.json";
const started = Date.now();

try {
  const res = await fetch(base + "/v1/bench", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "duo", suite: "tools", mode: "duo" }),
    signal: AbortSignal.timeout(5400000),
  });
  const body = await res.json();
  const out = {
    at: new Date().toISOString(),
    model: "duo",
    forced_mode: "duo",
    status: res.status,
    score: body.score ?? null,
    seconds: +((Date.now() - started) / 1000).toFixed(1),
    rows: (body.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds, answer: (r.answer || "").slice(0, 300), error: r.error })),
    error: body.error ?? null,
  };
  fs.writeFileSync(outFile, JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out).slice(0, 800));
} catch (e) {
  const out = { at: new Date().toISOString(), model: "duo", forced_mode: "duo", error: String(e).slice(0, 300), seconds: +((Date.now() - started) / 1000).toFixed(1) };
  fs.writeFileSync(outFile, JSON.stringify(out, null, 2));
  console.log("FAILED", JSON.stringify(out));
}
