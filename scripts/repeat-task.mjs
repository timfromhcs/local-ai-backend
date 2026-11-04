// Repeat one bench task N times per lane: a single run is not evidence.
// usage: node scripts/repeat-task.mjs <task-id> <runs> <model> [mode] [out-file]
import fs from "node:fs";

const [, , task = "hard-routing", runsArg = "5", model = "qwen3-4b", mode = "", outArg = ""] = process.argv;
const runs = Number(runsArg);
const base = "http://127.0.0.1:8090";
const outFile = outArg || `docs/repeat-${task}-${model}${mode ? "-" + mode : ""}.json`;
const rows = [];

for (let i = 1; i <= runs; i++) {
  const started = Date.now();
  const body = { model, suite: "hard", tasks: [task], max_tokens: 300 };
  if (mode) body.mode = mode;
  try {
    const res = await fetch(base + "/v1/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5400_000),
    });
    const payload = await res.json();
    const row = payload.rows?.[0] ?? {};
    rows.push({
      run: i, score: row.score ?? null, seconds: row.seconds ?? null,
      answer: (row.answer || "").slice(0, 160), error: row.error ?? payload.error ?? null,
      wall: +((Date.now() - started) / 1000).toFixed(1),
    });
  } catch (e) {
    rows.push({ run: i, score: null, error: String(e).slice(0, 200), wall: +((Date.now() - started) / 1000).toFixed(1) });
  }
  console.log(`run ${i}: score=${rows.at(-1).score} ${rows.at(-1).seconds ?? rows.at(-1).wall}s`);
}

const passes = rows.filter((r) => r.score === 1).length;
const summary = { at: new Date().toISOString(), task, model, mode: mode || "auto", runs, passes, rate: +(passes / runs).toFixed(2), rows };
fs.writeFileSync(outFile, JSON.stringify(summary, null, 2));
console.log(`${task} ${model}${mode ? " mode=" + mode : ""}: ${passes}/${runs} = ${summary.rate} -> ${outFile}`);
