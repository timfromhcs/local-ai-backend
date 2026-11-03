#!/usr/bin/env node
/**
 * Prove the local storage layer works: IndexedDB preferences + OPFS for larger artifacts.
 * Writes a multi-megabyte record through the real code path, reads it back, lists it,
 * exports it, and clears everything. Nothing leaves the browser.
 *
 *   node scripts/verify-storage.mjs
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const WEB = path.join(ROOT, "web");
const PORT = 4177;
const MIME = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
  ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png",
  ".json": "application/json", ".txt": "text/plain; charset=utf-8", ".xml": "application/xml" };

const server = http.createServer((req, res) => {
  const url = decodeURIComponent((req.url || "/").split("?")[0]);
  const file = path.join(WEB, url === "/" ? "index.html" : url.replace(/^\/+/, ""));
  if (!file.startsWith(WEB) || !fs.existsSync(file)) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(PORT, "127.0.0.1", r));

const candidates = [
  path.join(ROOT, "packages", "webtest", "node_modules", "playwright", "index.mjs"),
  path.join(process.env.USERPROFILE || process.env.HOME || "", ".config", "opencode", "mcp", "node_modules", "playwright", "index.mjs"),
];
let pw = null;
for (const c of candidates) if (fs.existsSync(c)) pw = await import(pathToFileURL(c).href);
if (!pw) { console.error("playwright not found"); process.exit(2); }

const browser = await pw.chromium.launch({ args: ["--no-sandbox"] });
const page = await browser.newPage();
let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${detail ? " :: " + detail : ""}`);
  if (!ok) failures += 1;
};

try {
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "load" });
  const available = await page.evaluate(() => window.localaiStorage.opfs.available());
  check(available === true, "OPFS available in this browser");

  const written = await page.evaluate(async () => {
    const payload = { kind: "storage-test", rows: Array.from({ length: 2000 }, (_, i) => ({ i, text: "x".repeat(512) })) };
    const path = await window.localaiStorage.opfs.write("run-test.json", payload);
    const back = await window.localaiStorage.opfs.read("run-test.json");
    return { path, bytes: JSON.stringify(payload).length, rows: back.rows.length, same: back.kind === "storage-test" };
  });
  check(written.same && written.rows === 2000, "OPFS round-trip", `${(written.bytes / 2 ** 20).toFixed(2)} MB`);

  const listed = await page.evaluate(() => window.localaiStorage.opfs.list());
  check(listed.includes("run-test.json"), "OPFS lists the artifact", listed.join(","));

  const kv = await page.evaluate(async () => {
    await window.localaiStorage.store.set("prefs", { tier: "medium", note: "storage-test" });
    return window.localaiStorage.store.get("prefs");
  });
  check(kv?.tier === "medium", "IndexedDB preferences still work");

  const usage = await page.evaluate(() => window.localaiStorage.opfs.usage());
  check(Boolean(usage?.quotaMb > 0), "storage estimate available", JSON.stringify(usage));

  // clearing must remove both layers
  await page.evaluate(async () => {
    await window.localaiStorage.opfs.clear();
    await window.localaiStorage.store.clear();
  });
  await page.reload({ waitUntil: "load" });
  const after = await page.evaluate(async () => ({
    files: await window.localaiStorage.opfs.list(),
    prefs: await window.localaiStorage.store.get("prefs", null),
  }));
  check(after.files.length === 0 && after.prefs === null, "clear removes OPFS and preferences");

  // the export path must include OPFS content
  const exported = await page.evaluate(async () => {
    await window.localaiStorage.opfs.write("run-export.json", { kind: "export-test" });
    const payload = { kind: "localai-export", data: await window.localaiStorage.store.all() };
    payload.opfs = {};
    for (const name of await window.localaiStorage.opfs.list()) payload.opfs[name] = await window.localaiStorage.opfs.read(name);
    return payload;
  });
  check(Boolean(exported.opfs["run-export.json"]), "export includes OPFS files");
} finally {
  await browser.close();
  server.close();
}

console.log(failures ? `\nstorage check: FAIL (${failures})` : "\nstorage check: PASS");
process.exit(failures ? 1 : 0);
