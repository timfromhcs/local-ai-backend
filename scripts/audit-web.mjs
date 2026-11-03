#!/usr/bin/env node
/**
 * Real accessibility audit for web/, measured in Chromium instead of asserted by hand.
 *
 * Checks: horizontal overflow (375/390/1440), tap targets >= 44px, image alt attributes,
 * heading structure, form labels, landmarks, visible focus, WCAG contrast for text, and
 * prefers-reduced-motion. Exits non-zero with a list of findings.
 *
 *   node scripts/audit-web.mjs [--port 4174] [--verbose]
 *
 * Playwright is resolved from packages/webtest/node_modules (CI) or the pinned OpenCode
 * install (local), so no new runtime dependency is invented for the demo.
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(import.meta.dirname, "..");
const WEB = path.join(ROOT, "web");
const PORT = Number(process.argv.includes("--port") ? process.argv[process.argv.indexOf("--port") + 1] : 4174);
const VERBOSE = process.argv.includes("--verbose");

const CANDIDATES = [
  path.join(ROOT, "packages", "webtest", "node_modules", "playwright", "index.mjs"),
  path.join(process.env.USERPROFILE || process.env.HOME || "", ".config", "opencode", "mcp", "node_modules", "playwright", "index.mjs"),
];

async function loadPlaywright() {
  for (const candidate of CANDIDATES) {
    if (fs.existsSync(candidate)) return import(pathToFileURL(candidate).href);
  }
  throw new Error(`playwright not found; looked in:\n  ${CANDIDATES.join("\n  ")}`);
}

const MIME = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
  ".svg": "image/svg+xml", ".json": "application/json", ".xml": "application/xml", ".txt": "text/plain; charset=utf-8",
  ".png": "image/png", ".webp": "image/webp", ".ico": "image/x-icon",
};

function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = path.join(WEB, url === "/" ? "index.html" : url.replace(/^\/+/, ""));
    if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(PORT, "127.0.0.1", () => resolve(server)));
}

const problems = [];
const note = (msg) => { problems.push(msg); console.log(`FAIL ${msg}`); };
const ok = (msg) => console.log(`ok   ${msg}`);

const CONTRAST_JS = `() => {
  const parse = (c) => {
    const m = c.match(/rgba?\\(([^)]+)\\)/);
    if (!m) return null;
    const p = m[1].split(",").map(Number);
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const lum = ({ r, g, b }) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m); return (x + 0.05) / (y + 0.05); };
  const bgOf = (el) => {
    for (let n = el; n; n = n.parentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c.a > 0.5) return c;
    }
    return { r: 11, g: 13, b: 18, a: 1 };
  };
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll("p, li, td, th, h1, h2, h3, summary, .note, .muted, label, button, a")) {
    const text = (el.textContent || "").trim();
    if (!text || seen.has(el)) continue;
    seen.add(el);
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const cs = getComputedStyle(el);
    const fg = parse(cs.color);
    if (!fg) continue;
    const size = parseFloat(cs.fontSize);
    const bold = Number(cs.fontWeight) >= 700;
    const large = size >= 24 || (bold && size >= 18.66);
    const r = ratio(fg, bgOf(el));
    const need = large ? 3 : 4.5;
    if (r + 0.01 < need) out.push({ text: text.slice(0, 48), ratio: Math.round(r * 100) / 100, need, size });
  }
  return out;
}`;

const TARGETS_JS = `() => {
  const sel = "button, a.btn, input, select, textarea, [role=tab]";
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.width < 44 || r.height < 44) {
      out.push({ tag: el.tagName.toLowerCase(), text: (el.textContent || el.id || "").trim().slice(0, 40), w: Math.round(r.width), h: Math.round(r.height) });
    }
  }
  return out;
}`;

const STRUCTURE_JS = `() => {
  const heads = [...document.querySelectorAll("h1,h2,h3,h4,h5,h6")].map((h) => Number(h.tagName[1]));
  const jumps = [];
  for (let i = 1; i < heads.length; i++) if (heads[i] - heads[i - 1] > 1) jumps.push(heads[i - 1] + "->" + heads[i]);
  const imgs = [...document.querySelectorAll("img")].filter((i) => !i.hasAttribute("alt")).map((i) => i.getAttribute("src"));
  const unlabeled = [...document.querySelectorAll("input,select,textarea")].filter((el) => {
    if (el.type === "hidden" || el.hidden) return false;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;  // visually hidden utility fields
    if (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby")) return false;
    return !(el.id && document.querySelector('label[for="' + el.id + '"]'));
  }).map((el) => el.tagName.toLowerCase() + (el.id ? "#" + el.id : ""));
  return {
    h1: heads.filter((h) => h === 1).length,
    jumps,
    imgs,
    unlabeled,
    landmarks: ["header", "main", "nav", "footer"].filter((t) => !document.querySelector(t)),
    lang: document.documentElement.lang,
  };
}`;

const server = await serve();
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ args: ["--no-sandbox"] });

try {
  for (const width of [375, 390, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "load" });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => ({
      scrollW: document.documentElement.scrollWidth,
      clientW: document.documentElement.clientWidth,
    }));
    if (overflow.scrollW > overflow.clientW + 1) note(`${width}px: horizontal overflow ${overflow.scrollW} > ${overflow.clientW}`);
    else ok(`${width}px: no horizontal overflow`);

    if (width === 390) {
      const targets = await page.evaluate(new Function(`return (${TARGETS_JS})()`));
      for (const t of targets) note(`tap target < 44px: ${t.tag} "${t.text}" ${t.w}x${t.h}`);
      if (!targets.length) ok("tap targets >= 44px");

      const contrast = await page.evaluate(new Function(`return (${CONTRAST_JS})()`));
      for (const c of contrast) note(`contrast ${c.ratio} < ${c.need} (${c.size}px): "${c.text}"`);
      if (!contrast.length) ok("text contrast passes WCAG AA");
    }

    const structure = await page.evaluate(new Function(`return (${STRUCTURE_JS})()`));
    if (structure.h1 !== 1) note(`${width}px: expected exactly one h1, found ${structure.h1}`);
    if (structure.jumps.length) note(`${width}px: heading level jumps ${structure.jumps.join(",")}`);
    for (const src of structure.imgs) note(`${width}px: <img> without alt: ${src}`);
    for (const field of structure.unlabeled) note(`${width}px: form field without label: ${field}`);
    if (structure.landmarks.length) note(`${width}px: missing landmarks ${structure.landmarks.join(",")}`);
    if (!structure.jumps.length && !structure.imgs.length && !structure.unlabeled.length && !structure.landmarks.length) {
      ok(`${width}px: structure, labels, landmarks`);
    }

    if (width === 390) {
      // keyboard focus must be visible
      await page.keyboard.press("Tab");
      const focus = await page.evaluate(() => {
        const el = document.activeElement;
        const cs = el ? getComputedStyle(el) : null;
        return el ? {
          tag: el.tagName.toLowerCase(),
          outline: cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0,
          shadow: cs.boxShadow !== "none",
        } : null;
      });
      if (!focus || (!focus.outline && !focus.shadow)) note("first Tab stop has no visible focus indicator");
      else ok(`visible focus on ${focus.tag}`);

      // reduced motion must actually disable transitions
      await page.emulateMedia({ reducedMotion: "reduce" });
      const motion = await page.evaluate(() => {
        const el = document.querySelector(".btn");
        const cs = getComputedStyle(el);
        return { duration: cs.transitionDuration, animation: cs.animationName };
      });
      if (parseFloat(motion.duration) > 0 || motion.animation !== "none") note(`reduced-motion not honoured (${JSON.stringify(motion)})`);
      else ok("prefers-reduced-motion disables motion");

      // PWA: manifest reachable and a service worker actually registers
      const manifest = await page.evaluate(() => document.querySelector('link[rel="manifest"]')?.href ?? null);
      if (!manifest) note("no manifest link");
      else {
        const res = await page.evaluate(async (href) => {
          const r = await fetch(href);
          const json = await r.json().catch(() => null);
          return { status: r.status, display: json?.display ?? null };
        }, manifest);
        if (res.status !== 200 || res.display !== "standalone") note(`manifest not usable (${JSON.stringify(res)})`);
        else ok("manifest reachable and standalone");
      }

      const sw = await page.evaluate(async () => {
        if (!("serviceWorker" in navigator)) return { supported: false };
        const reg = await navigator.serviceWorker.getRegistration();
        return { supported: true, scope: reg?.scope ?? null, active: Boolean(reg?.active) };
      });
      if (!sw.supported) note("service worker unsupported");
      else if (!sw.active) note("service worker did not activate");
      else ok(`service worker active (${sw.scope})`);
    }
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}

if (problems.length) {
  console.log(`\n${problems.length} accessibility finding(s)`);
  process.exit(1);
}
console.log("\naccessibility audit: PASS");
