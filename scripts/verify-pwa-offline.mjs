#!/usr/bin/env node
/**
 * Prove the PWA actually works offline: load the page, wait for the service worker, cut the
 * network, reload, and check that the shell still renders with CSS applied.
 *
 *   node scripts/verify-pwa-offline.mjs
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const WEB = path.join(ROOT, "web");
const PORT = 4176;
const MIME = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
  ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json",
  ".txt": "text/plain; charset=utf-8", ".xml": "application/xml" };

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
await new Promise((resolve) => server.listen(PORT, "127.0.0.1", resolve));

const candidates = [
  path.join(ROOT, "packages", "webtest", "node_modules", "playwright", "index.mjs"),
  path.join(process.env.USERPROFILE || process.env.HOME || "", ".config", "opencode", "mcp", "node_modules", "playwright", "index.mjs"),
];
let pw = null;
for (const candidate of candidates) if (fs.existsSync(candidate)) pw = await import(pathToFileURL(candidate).href);
if (!pw) {
  console.error("playwright not found");
  process.exit(2);
}

const browser = await pw.chromium.launch({ args: ["--no-sandbox"] });
const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
const page = await context.newPage();
let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${detail ? " :: " + detail : ""}`);
  if (!ok) failures += 1;
};

try {
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "load" });
  await page.evaluate(() => navigator.serviceWorker.ready);
  const sw = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.getRegistration();
    return { active: Boolean(reg?.active), state: reg?.active?.state ?? null };
  });
  check(sw.active && sw.state === "activated", "service worker activated", sw.state ?? "none");

  const manifest = await page.evaluate(async () => {
    const href = document.querySelector('link[rel="manifest"]')?.href;
    if (!href) return null;
    const r = await fetch(href);
    const j = await r.json();
    return { ok: r.ok, display: j.display, icons: j.icons?.length ?? 0 };
  });
  check(Boolean(manifest?.ok) && manifest.display === "standalone", "manifest loads", JSON.stringify(manifest));

  // give the worker a moment to finish precaching the shell
  await page.waitForTimeout(1200);

  await context.setOffline(true);
  await page.reload({ waitUntil: "domcontentloaded" });
  const offline = await page.evaluate(() => {
    const h1 = document.querySelector("h1")?.textContent?.trim() ?? "";
    const bg = getComputedStyle(document.body).backgroundColor;
    return { h1, bg, sections: document.querySelectorAll("main section").length };
  });
  check(offline.h1.length > 5, "shell renders offline", offline.h1.slice(0, 40));
  check(offline.sections >= 8, "all sections present offline", `${offline.sections} sections`);
  check(offline.bg !== "rgba(0, 0, 0, 0)" && offline.bg !== "rgb(255, 255, 255)", "stylesheet cached", offline.bg);

  // the local gateway must NOT be served from cache
  const apiBypass = await page.evaluate(async (port) => {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/healthz`, { cache: "no-store" });
      return { status: r.status };
    } catch (e) {
      return { error: String(e).slice(0, 40) };
    }
  }, PORT);
  check(Boolean(apiBypass.error) || apiBypass.status >= 400, "gateway requests are not served offline", JSON.stringify(apiBypass));
} finally {
  await browser.close();
  server.close();
}

console.log(failures ? `\npwa offline check: FAIL (${failures})` : "\npwa offline check: PASS");
process.exit(failures ? 1 : 0);
