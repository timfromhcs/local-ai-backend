// Measure the tools suite for duo (plan/execute/verify) and persist the result.
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const model = process.argv[2] || "duo";
const outFile = process.argv[3] || `docs/tools-bench-${model}.json`;
const started = Date.now();

try {
  const res = await fetch(base + "/v1/bench", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, suite: "tools" }),
    signal: AbortSignal.timeout(5400000),
  });
  const body = await res.json();
  const out = {
    at: new Date().toISOString(),
    model,
    status: res.status,
    score: body.score ?? null,
    seconds: +((Date.now() - started) / 1000).toFixed(1),
    rows: (body.rows ?? []).map((r) => ({
      id: r.id, score: r.score, seconds: r.seconds,
      answer: (r.answer || "").slice(0, 200), error: r.error,
    })),
    error: body.error ?? null,
  };
  fs.writeFileSync(outFile, JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out).slice(0, 600));
} catch (e) {
  const out = { at: new Date().toISOString(), model, error: String(e).slice(0, 300), seconds: +((Date.now() - started) / 1000).toFixed(1) };
  fs.writeFileSync(outFile, JSON.stringify(out, null, 2));
  console.log("FAILED", JSON.stringify(out));
}
