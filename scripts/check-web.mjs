#!/usr/bin/env node
/**
 * Static gate for the web demo. No browser needed, so CI stays fast:
 *  - vercel.json parses and pins the "Other" preset (framework: null)
 *  - every JS file is valid syntax
 *  - every i18n key used by index.html/app.js exists in EN, DE and FR
 *  - no locale defines keys nobody uses (drift in the other direction)
 *  - required DOM ids/asset files exist
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const web = path.join(root, "web");
const problems = [];
const ok = (msg) => console.log(`ok   ${msg}`);
const fail = (msg) => { problems.push(msg); console.log(`FAIL ${msg}`); };

// 1. vercel.json
let cfg = null;
try {
  cfg = JSON.parse(fs.readFileSync(path.join(web, "vercel.json"), "utf8"));
  ok("vercel.json parses");
} catch (e) {
  fail(`vercel.json: ${e.message}`);
}
if (cfg) {
  if (cfg.framework !== null) fail("vercel.json must set \"framework\": null (Other preset)");
  else ok("vercel.json pins framework: null");
  if (cfg.outputDirectory !== ".") fail(`vercel.json outputDirectory should be "." (got ${cfg.outputDirectory})`);
  else ok("vercel.json outputDirectory is .");
}

// 2. JS syntax
const jsFiles = ["app.js", "i18n.js"].map((f) => path.join(web, f));
for (const f of jsFiles) {
  try {
    execFileSync(process.execPath, ["--check", f], { stdio: "pipe" });
    ok(`${path.basename(f)} syntax`);
  } catch (e) {
    fail(`${path.basename(f)} syntax: ${String(e.stderr || e.message).slice(0, 300)}`);
  }
}

// 3. i18n coverage
const i18nSrc = fs.readFileSync(path.join(web, "i18n.js"), "utf8");
function localeKeys(name) {
  const start = i18nSrc.indexOf(`const ${name} = {`);
  if (start < 0) return null;
  const end = i18nSrc.indexOf("\n  };", start);
  const body = i18nSrc.slice(start, end);
  return new Set([...body.matchAll(/'([a-zA-Z0-9._-]+)':/g)].map((m) => m[1]));
}
const locales = Object.fromEntries(["en", "de", "fr"].map((n) => [n, localeKeys(n)]));
for (const [name, keys] of Object.entries(locales)) {
  if (!keys) fail(`locale ${name} not found in i18n.js`);
  else ok(`locale ${name}: ${keys.size} keys`);
}

const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
const appjs = fs.readFileSync(path.join(web, "app.js"), "utf8");
const used = new Set([
  ...[...html.matchAll(/data-i18n(?:-ph)?="([^"]+)"/g)].map((m) => m[1]),
  ...[...appjs.matchAll(/\bt\('([^']+)'/g)].map((m) => m[1]),
]);

const en = locales.en ?? new Set();
for (const key of used) {
  if (!en.has(key)) fail(`used but missing in EN: ${key}`);
  for (const loc of ["de", "fr"]) {
    if (!locales[loc]?.has(key)) fail(`used but missing in ${loc.toUpperCase()}: ${key}`);
  }
}
if (problems.length === 0) ok(`${used.size} i18n keys used, present in EN/DE/FR`);

for (const key of en) {
  if (!used.has(key)) fail(`defined but unused (drift): ${key}`);
}

// 4. required DOM structure and assets
for (const id of ["demo", "lab", "guide", "numbers", "roadmap", "run-lab", "download-lab", "load-model", "probe"]) {
  if (!html.includes(`id="${id}"`)) fail(`index.html missing #${id}`);
}
for (const asset of ["logo.svg", "hero.svg", "diagram.svg", "styles.css"]) {
  if (!fs.existsSync(path.join(web, "assets", asset)) && asset.endsWith(".svg")) fail(`missing asset ${asset}`);
}
for (const f of ["index.html", "styles.css", "app.js", "i18n.js", "vercel.json"]) {
  if (!fs.existsSync(path.join(web, f))) fail(`missing web/${f}`);
}
if (problems.length === 0) ok("DOM anchors and assets present");

// 5. no external stylesheet/font request that would break offline checks
if (/<link[^>]+rel="stylesheet"[^>]+href="https?:\/\//.test(html)) fail("index.html loads an external stylesheet");
else ok("no external stylesheet");

// 6. SEO and discovery
const head = html.slice(0, html.indexOf("</head>"));
for (const needle of ['rel="canonical"', 'hreflang="en"', 'hreflang="de"', 'hreflang="fr"', 'hreflang="x-default"', 'application/ld+json']) {
  if (!head.includes(needle)) fail(`index.html head missing ${needle}`);
}
const ldBlocks = [...head.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (ldBlocks.length < 2) fail("expected SoftwareApplication + FAQPage JSON-LD");
for (const block of ldBlocks) {
  try {
    const parsed = JSON.parse(block);
    if (!parsed["@context"].includes("schema.org")) fail("JSON-LD without schema.org context");
  } catch (e) {
    fail(`JSON-LD does not parse: ${e.message}`);
  }
}
for (const file of ["robots.txt", "sitemap.xml"]) {
  if (!fs.existsSync(path.join(web, file))) fail(`missing web/${file}`);
}
const sitemap = fs.existsSync(path.join(web, "sitemap.xml")) ? fs.readFileSync(path.join(web, "sitemap.xml"), "utf8") : "";
if (sitemap && !sitemap.includes("https://local-ai-backend-blue.vercel.app/")) fail("sitemap lacks the production URL");
if (sitemap && !/hreflang="(en|de|fr)"/.test(sitemap)) fail("sitemap lacks hreflang alternates");
if (problems.length === 0) ok("SEO: canonical, hreflang, JSON-LD, robots, sitemap");

// 6. security headers: a policy that cannot silently rot
const securityHeaders = (cfg.headers ?? [])
  .flatMap((rule) => rule.headers ?? [])
  .reduce((acc, h) => ({ ...acc, [h.key.toLowerCase()]: h.value }), {});
for (const required of ["content-security-policy", "x-content-type-options", "referrer-policy", "permissions-policy", "strict-transport-security"]) {
  if (!securityHeaders[required]) fail(`vercel.json missing header: ${required}`);
}
const csp = securityHeaders["content-security-policy"] ?? "";
if (csp) {
  if (/\bunsafe-eval\b/.test(csp)) fail("CSP allows unsafe-eval");
  if (/script-src[^;]*'unsafe-inline'/.test(csp)) fail("CSP allows inline scripts");
  if (!/script-src[^;]*['"]?self/.test(csp)) fail("CSP script-src must allow same-origin scripts");
  if (!/default-src 'self'/.test(csp)) fail("CSP default-src should be 'self'");
  if (!/object-src 'none'/.test(csp)) fail("CSP should disable object-src");
  if (!/frame-ancestors 'none'/.test(csp)) fail("CSP should set frame-ancestors 'none'");
  ok(`CSP present (${csp.length} chars), no unsafe-eval, no inline scripts`);
}
if (!securityHeaders["x-frame-options"]) fail("missing X-Frame-Options");

// 7. PWA: manifest, service worker and the link that makes them discoverable
const manifestPath = path.join(web, "manifest.webmanifest");
if (!fs.existsSync(manifestPath)) fail("missing web/manifest.webmanifest");
else {
  try {
    const m = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    for (const key of ["name", "short_name", "start_url", "display", "background_color", "theme_color", "icons"]) {
      if (!m[key]) fail(`manifest missing ${key}`);
    }
    const sizes = (m.icons ?? []).map((i) => i.sizes);
    if (!sizes.includes("192x192")) fail("manifest needs a 192x192 icon");
    if (!sizes.includes("512x512")) fail("manifest needs a 512x512 icon");
    if (!(m.icons ?? []).some((i) => i.purpose === "maskable")) fail("manifest needs a maskable icon");
    for (const icon of m.icons ?? []) {
      if (!fs.existsSync(path.join(web, icon.src.replace(/^\//, "")))) fail(`manifest icon missing: ${icon.src}`);
    }
    ok(`manifest valid (${m.icons.length} icons, display ${m.display})`);
  } catch (e) {
    fail(`manifest does not parse: ${e.message}`);
  }
}
if (!head.includes('rel="manifest"')) fail("index.html does not link the manifest");
if (!head.includes("apple-touch-icon")) fail("index.html does not link an apple-touch-icon");

const swPath = path.join(web, "sw.js");
if (!fs.existsSync(swPath)) fail("missing web/sw.js");
else {
  const sw = fs.readFileSync(swPath, "utf8");
  for (const needed of ["addEventListener(\"install\"", "addEventListener(\"fetch\"", '"/index.html"']) {
    if (!sw.includes(needed)) fail(`sw.js missing ${needed}`);
  }
  if (/\.gguf|huggingface/.test(sw)) fail("the service worker must not cache model weights");
  ok("service worker precaches the shell and ignores models");
}

console.log(problems.length ? `\n${problems.length} problem(s)` : "\nweb check: PASS");
process.exit(problems.length ? 1 : 0);
