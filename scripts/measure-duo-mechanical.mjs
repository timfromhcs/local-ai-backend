// Duo with a mechanical verdict on the code suite: does a red test run beat the LLM verifier?
import fs from "node:fs";

const base = "http://127.0.0.1:8090";
const out = { at: new Date().toISOString(), suite: "code", runs: [] };
const file = process.argv[2] || "docs/duo-mechanical.json";

for (const i of [1, 2]) {
  const t0 = Date.now();
  try {
    const res = await fetch(base + "/v1/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "duo", suite: "code", max_tokens: 600, mode: "duo" }),
      signal: AbortSignal.timeout(5400_000),
    });
    const b = await res.json();
    out.runs.push({ run: i, model: "duo", lane: "duo", status: res.status, score: b.score ?? null,
                    seconds: +((Date.now() - t0) / 1000).toFixed(1),
                    rows: (b.rows ?? []).map((r) => ({ id: r.id, score: r.score, seconds: r.seconds,
                                                       answer: (r.answer || "").slice(0, 400) })),
                    error: b.error ?? null });
  } catch (e) {
    out.runs.push({ run: i, model: "duo", lane: "duo", error: String(e).slice(0, 300),
                    seconds: +((Date.now() - t0) / 1000).toFixed(1) });
  }
  console.log("RUN", JSON.stringify(out.runs.at(-1)));
}

fs.writeFileSync(file, JSON.stringify(out, null, 2));
console.log("written", file);
