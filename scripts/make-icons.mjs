#!/usr/bin/env node
/**
 * Render web/assets/logo.svg into the PNG sizes a PWA needs (192, 512, maskable 512,
 * apple-touch 180). Uses the local Chromium, so no image library and no network.
 *
 *   node scripts/make-icons.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const OUT = path.join(ROOT, "web", "assets");
const svg = fs.readFileSync(path.join(OUT, "logo.svg"), "utf8");

const CANDIDATES = [
  path.join(ROOT, "packages", "webtest", "node_modules", "playwright", "index.mjs"),
  path.join(process.env.USERPROFILE || process.env.HOME || "", ".config", "opencode", "mcp", "node_modules", "playwright", "index.mjs"),
];

async function loadPlaywright() {
  for (const candidate of CANDIDATES) if (fs.existsSync(candidate)) return import(pathToFileURL(candidate).href);
  throw new Error("playwright not found");
}

const TARGETS = [
  { file: "icon-192.png", size: 192, padding: 0.06 },
  { file: "icon-512.png", size: 512, padding: 0.06 },
  { file: "icon-maskable-512.png", size: 512, padding: 0.2 }, // safe zone for Android masks
  { file: "apple-touch-icon.png", size: 180, padding: 0.08 },
];

const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ args: ["--no-sandbox"] });
try {
  for (const { file, size, padding } of TARGETS) {
    const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
    const pad = Math.round(size * padding);
    await page.setContent(`<!doctype html><html><body style="margin:0;width:${size}px;height:${size}px;background:#0b0d12;display:grid;place-items:center">
      <div style="width:${size - pad * 2}px;height:${size - pad * 2}px;display:grid;place-items:center">${svg}</div>
    </body></html>`);
    const buffer = await page.screenshot({ type: "png", omitBackground: false });
    fs.writeFileSync(path.join(OUT, file), buffer);
    console.log("wrote", file, `${buffer.length} bytes`);
    await page.close();
  }
} finally {
  await browser.close();
}
