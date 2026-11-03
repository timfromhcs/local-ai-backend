#!/usr/bin/env node
/**
 * rsi-capture-mcp — record what actually happened, then derive training pairs.
 *
 * The gateway owns the logic (scrubbing, verified-only, synthetic marking); this server is a
 * thin MCP surface over /v1/rsi so any agent can capture its own work:
 *
 *   rsi_start   -> open a session
 *   rsi_log     -> append an event (user / assistant / tool / verification / note)
 *   rsi_finalize-> derive pairs for verified sessions and write *.pairs.jsonl
 *   rsi_pairs   -> inspect the derived pairs
 *   rsi_stats   -> sessions, verified sessions, pairs, events, synthetic steps
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const BASE = (process.env.LOCALAI_GATEWAY_URL || "http://127.0.0.1:8090").replace(/\/$/, "");

const text = (v) => ({ content: [{ type: "text", text: typeof v === "string" ? v : JSON.stringify(v, null, 2) }] });
const fail = (e) => ({ isError: true, content: [{ type: "text", text: String(e?.message || e).slice(0, 600) }] });

async function call(path, method = "GET", body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30000),
  });
  const raw = await res.text();
  let data;
  try { data = JSON.parse(raw); } catch { data = raw; }
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${JSON.stringify(data).slice(0, 300)}`);
  return data;
}

const server = new Server(
  { name: "rsi-capture-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "rsi_start",
      description: `Open (or resume) a capture session on the gateway at ${BASE}. Call this before logging events.`,
      inputSchema: {
        type: "object",
        properties: { session: { type: "string", description: "Stable id, e.g. rsi-2026-11-03-authfix" } },
        required: ["session"],
      },
    },
    {
      name: "rsi_log",
      description:
        "Append one event. Use type 'tool' for executed commands/edits with ok true/false, type 'verification' only when you can point at a passing test or command. Mark invented intermediate steps with synthetic: true - they are counted, never presented as executed.",
      inputSchema: {
        type: "object",
        properties: {
          session: { type: "string" },
          type: { type: "string", enum: ["user", "assistant", "tool", "verification", "note"] },
          content: { type: "string", description: "For user/assistant/note" },
          name: { type: "string", description: "Tool name for type=tool" },
          input: { type: "object", description: "Tool arguments" },
          result: { type: "string", description: "Tool output" },
          ok: { type: "boolean", description: "tool: succeeded? verification: passed?" },
          detail: { type: "string", description: "verification evidence, e.g. 'pytest 5 passed'" },
          synthetic: { type: "boolean", description: "true when the step was not executed" },
        },
        required: ["session", "type"],
      },
    },
    {
      name: "rsi_finalize",
      description: "Derive training pairs for a session. Only sessions with a successful verification produce pairs.",
      inputSchema: { type: "object", properties: { session: { type: "string" } }, required: ["session"] },
    },
    {
      name: "rsi_pairs",
      description: "Show the derived chat-format pairs (scrubbed, retries collapsed, meta.verified required).",
      inputSchema: { type: "object", properties: { session: { type: "string" } }, required: ["session"] },
    },
    {
      name: "rsi_stats",
      description: "Sessions, verified sessions, pairs, events and synthetic steps currently on disk.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  try {
    if (name === "rsi_start") return text(await call("/v1/rsi/sessions", "POST", { session: args.session }));
    if (name === "rsi_log") {
      const { session, ...event } = args;
      return text(await call("/v1/rsi/events", "POST", { session, ...event }));
    }
    if (name === "rsi_finalize") return text(await call("/v1/rsi/finalize", "POST", { session: args.session }));
    if (name === "rsi_pairs") return text(await call(`/v1/rsi/pairs?session=${encodeURIComponent(args.session)}`));
    if (name === "rsi_stats") return text(await call("/v1/rsi/stats"));
    throw new Error(`unknown tool ${name}`);
  } catch (e) {
    return fail(e);
  }
});

server.onerror = (e) => console.error("[rsi-capture-mcp]", e);
await server.connect(new StdioServerTransport());
