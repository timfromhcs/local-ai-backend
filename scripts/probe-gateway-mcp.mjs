#!/usr/bin/env node
/**
 * Probe the localai-gateway-mcp server with the official MCP client.
 * Fails when the server does not answer or the gateway it points at is down,
 * so it can run in CI as a real integration check.
 *
 *   node scripts/probe-gateway-mcp.mjs [gatewayUrl]
 */
import path from "node:path";
import { pathToFileURL } from "node:url";

const gateway = process.argv[2] || "http://127.0.0.1:8090";
const serverDir = path.resolve(import.meta.dirname, "..", "servers", "localai-gateway-mcp");
const sdk = (p) => pathToFileURL(path.join(serverDir, "node_modules", "@modelcontextprotocol", "sdk", "dist", "esm", p)).href;

const { Client } = await import(sdk("client/index.js"));
const { StdioClientTransport } = await import(sdk("client/stdio.js"));
const server = path.join(serverDir, "index.js");

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [server],
  env: { ...process.env, LOCALAI_GATEWAY_URL: gateway },
});
const client = new Client({ name: "gateway-mcp-probe", version: "1.0.0" }, { capabilities: {} });

try {
  await client.connect(transport);
  const { tools } = await client.listTools();
  const names = tools.map((t) => t.name).sort();
  console.log(`connected: ${names.join(", ")}`);

  const expected = ["benchmark", "chat", "doctor", "gateway_status", "list_models"];
  const missing = expected.filter((n) => !names.includes(n));
  if (missing.length) throw new Error(`missing tools: ${missing.join(", ")}`);

  const status = await client.callTool({ name: "gateway_status", arguments: {} });
  const text = status.content?.[0]?.text ?? "";
  console.log("gateway_status:", text.replace(/\s+/g, " ").slice(0, 200));
  if (status.isError) throw new Error(text);

  console.log("gateway-mcp probe: PASS");
  await client.close();
  process.exit(0);
} catch (e) {
  console.error("gateway-mcp probe: FAIL -", String(e.message || e).slice(0, 400));
  process.exit(1);
}
