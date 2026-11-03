#!/usr/bin/env node
/**
 * Verify the DEPLOYED site, not the local files: manifest, service worker, and an offline
 * reload with the network cut. Usage: node scripts/verify-pwa-production.mjs [url]
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const url = process.argv[2] || "https://local-ai-backend-blue.vercel.app";
const ROOT = path.resolve(import.meta.dirname, "..");
const candidates = [
  path.join(ROOT, "packages", "webtest", "node_modules", "playwright", "index.mjs"),
  path.join(process.env.USERPROFILE || process.env.HOME || "", ".config", "opencode", "mcp", "node_modules", "playwright", "index.mjs"),
];
let pw = null;
for (const c of candidates) if (fs.existsSync(c)) pw = await import(pathToFileURL(c).href);
if (!pw) { console.error("playwright not found"); process.exit(2); }

const browser = await pw.chromium.launch({ args: ["--no-sandbox"] });
const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
const page = await context.newPage();
let failures = 0;
const check = (ok, label, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${detail ? " :: " + detail : ""}`);
  if (!ok) failures += 1;
};

try {
  const response = await page.goto(url, { waitUntil: "load" });
  check(response?.status() === 200, "page loads", String(response?.status()));
  const csp = response?.headers()["content-security-policy"] ?? "";
  check(/default-src 'self'/.test(csp) && !/unsafe-eval/.test(csp), "CSP present and strict", csp.slice(0, 60));

  await page.evaluate(() => navigator.serviceWorker.ready);
  const sw = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.getRegistration();
    return { active: Boolean(reg?.active), state: reg?.active?.state ?? null, scope: reg?.scope ?? null };
  });
  check(sw.active && sw.state === "activated", "service worker activated", sw.scope ?? "none");

  const manifest = await page.evaluate(async () => {
    const href = document.querySelector('link[rel="manifest"]')?.href;
    if (!href) return null;
    const r = await fetch(href);
    const j = await r.json();
    return { ok: r.ok, display: j.display, icons: j.icons?.length ?? 0, name: j.name };
  });
  check(Boolean(manifest?.ok) && manifest.display === "standalone", "manifest valid", JSON.stringify(manifest));

  await page.waitForTimeout(2000); // let the precache finish
  await context.setOffline(true);
  await page.reload({ waitUntil: "domcontentloaded" });
  const offline = await page.evaluate(() => ({
    h1: document.querySelector("h1")?.textContent?.trim() ?? "",
    sections: document.querySelectorAll("main section").length,
    bg: getComputedStyle(document.body).backgroundColor,
  }));
  check(offline.h1.length > 5 && offline.sections >= 8, "shell renders offline", `${offline.sections} sections`);
  check(offline.bg === "rgb(11, 13, 18)", "stylesheet served from cache", offline.bg);
} finally {
  await browser.close();
}
console.log(failures ? `\nproduction PWA: FAIL (${failures})` : "\nproduction PWA: PASS");
process.exit(failures ? 1 : 0);
