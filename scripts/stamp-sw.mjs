#!/usr/bin/env node
/**
 * Stamp the service worker cache name before a deploy.
 *
 * Without a build step, stale-while-revalidate can serve the previous app.js on the first load
 * after a deploy (observed on production). Changing SW_VERSION makes the new worker re-cache the
 * shell immediately, so the first load after a deploy is current.
 *
 *   node scripts/stamp-sw.mjs [--check]
 */
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const SW = path.join(ROOT, "web", "sw.js");
const source = fs.readFileSync(SW, "utf8");
const match = source.match(/const SW_VERSION = "([^"]+)";/);
if (!match) {
  console.error("SW_VERSION not found in web/sw.js");
  process.exit(1);
}

const stamp = `v${Math.floor(Date.now() / 1000)}`;
if (process.argv.includes("--check")) {
  if (/^v\d{10}$/.test(match[1])) {
    console.log(`sw cache version is stamped (${match[1]})`);
    process.exit(0);
  }
  console.log(`sw cache version is not stamped (${match[1]}); run: node scripts/stamp-sw.mjs`);
  process.exit(1);
}

const updated = source.replace(match[0], `const SW_VERSION = "${stamp}";`);
fs.writeFileSync(SW, updated);
console.log(`stamped web/sw.js: ${match[1]} -> ${stamp}`);
