#!/usr/bin/env node
// Download the LocalAI Gateway binary once, verify it, then never touch the network again.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const pkg = require("../package.json");
const vendor = path.join(__dirname, "..", "vendor");
const exe = path.join(vendor, process.platform === "win32" ? "localai-gateway.exe" : "localai-gateway");
const repo = "timfromhcs/local-ai-backend";

function log(msg) {
  console.log(`[localai-gateway] ${msg}`);
}

function skip(reason) {
  log(`skipping download: ${reason}`);
  log("install manually from https://github.com/timfromhcs/local-ai-backend/releases");
  process.exit(0); // never break an install because GitHub is unreachable
}

async function main() {
  if (process.env.LOCALAI_SKIP_DOWNLOAD === "1") return skip("LOCALAI_SKIP_DOWNLOAD=1");
  if (process.platform !== "win32") return skip(`no prebuilt binary for ${process.platform}`);
  if (fs.existsSync(exe)) return log(`binary already present at ${exe}`);

  const tag = `v${pkg.version}`;
  const base = `https://github.com/${repo}/releases/download/${tag}`;
  fs.mkdirSync(vendor, { recursive: true });

  log(`downloading ${tag} ...`);
  const bin = Buffer.from(await (await fetch(`${base}/localai-gateway.exe`)).arrayBuffer());
  const sums = await (await fetch(`${base}/SHA256SUMS.txt`)).text();
  const expected = (sums.split(/\r?\n/).find((l) => l.includes("localai-gateway.exe")) || "").split(/\s+/)[0];
  const actual = crypto.createHash("sha256").update(bin).digest("hex");

  if (!expected || expected !== actual) {
    log(`checksum mismatch (expected ${expected || "?"}, got ${actual}) - deleting download`);
    process.exit(0);
  }
  fs.writeFileSync(exe, bin);
  try { fs.chmodSync(exe, 0o755); } catch {}
  log(`installed ${exe} (sha256 ${actual.slice(0, 12)}...)`);
}

main().catch((e) => skip(String(e.message || e)));
