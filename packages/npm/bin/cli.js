#!/usr/bin/env node
// Thin launcher: forward every argument to the verified binary.
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const exe = path.join(__dirname, "..", "vendor", process.platform === "win32" ? "localai-gateway.exe" : "localai-gateway");

if (!fs.existsSync(exe)) {
  console.error("localai-gateway binary not installed.");
  console.error("Run: npm rebuild @hcsmedia/localai-gateway");
  console.error("Or download it from https://github.com/timfromhcs/local-ai-backend/releases");
  process.exit(1);
}

const child = spawn(exe, process.argv.slice(2), { stdio: "inherit" });
child.on("exit", (code, signal) => process.exit(signal ? 1 : code ?? 0));
child.on("error", (e) => {
  console.error(`failed to start ${exe}: ${e.message}`);
  process.exit(1);
});
