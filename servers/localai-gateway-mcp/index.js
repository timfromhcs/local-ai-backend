#!/usr/bin/env node
/**
 * localai-gateway-mcp — MCP surface for a running LocalAI Gateway.
 *
 * Everything goes through the gateway's OpenAI-compatible HTTP API, so this server
 * adds no inference code of its own. Point it at any instance:
 *   LOCALAI_GATEWAY_URL=http://127.0.0.1:8090
 *
 * Tools: gateway_status, list_models, chat, benchmark, doctor
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const BASE = (process.env.LOCALAI_GATEWAY_URL || "http://127.0.0.1:8090").replace(/\/$/, "");

const text = (v) => ({ content: [{ type: "text", text: typeof v === "string" ? v : JSON.stringify(v, null, 2) }] });
const fail = (e) => ({ isError: true, content: [{ type: "text", text: String(e?.message || e).slice(0, 600) }] });

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

async function post(path, body, timeoutMs = 300000) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const raw = await res.text();
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${raw.slice(0, 300)}`);
  return JSON.parse(raw);
}

/* deterministic benchmark, mirrors bench/run.py (no LLM judge) */
const BENCH = [
  ["routing", "A task needs planning, implementation and verification. Answer with one word: fast, duo or bigout.", (a) => (/\bduo\b/i.test(a) ? 1 : 0)],
  ["routing-hard", "Migrate three services, keep every test green and prove rollback. One word: fast, duo or bigout.", (a) => (/\bbigout\b/i.test(a) ? 1 : 0)],
  ["tool-call", "List the files in C:\\work, then read package.json. Reply with the tool calls only.", (a) => (/read_file\s*\(/i.test(a) && /(list_files?|list_dir)\s*\(/i.test(a) ? 1 : 0)],
  ["json", 'Return only JSON: {"model":"duo","loaded":true,"ram_mb":9000}', (a) => {
    try {
      const m = a.match(/\{[\s\S]*\}/);
      const j = JSON.parse(m ? m[0] : a);
      return j.model === "duo" && j.loaded === true ? 1 : 0;
    } catch { return 0; }
  }],
  ["math", "A model decodes 5.9 tokens per second. How many tokens in 10 minutes? Number only.", (a) => (/\b3540\b/.test(a.replace(/[.,]/g, "")) ? 1 : 0)],
  ["honesty", "Can a browser run a 27B model at 50 tokens per second? Answer No or Yes, then one short reason.", (a) => (/^\s*no\b/i.test(a) ? 1 : 0)],
];

async function chatCompletion(model, content, maxTokens = 300) {
  const data = await post("/v1/chat/completions", {
    model,
    messages: [{ role: "user", content }],
    max_tokens: maxTokens,
    temperature: 0,
  });
  const m = data.choices?.[0]?.message ?? {};
  const c = m.content;
  return Array.isArray(c) ? c.map((b) => b.text ?? "").join(" ") : c ?? "";
}

const server = new Server(
  { name: "localai-gateway-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "gateway_status",
      description: `Health and model summary of the gateway at ${BASE}. Use this before anything else.`,
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "list_models",
      description: "List every registered model with kind, status, runtime and whether it is loaded.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "chat",
      description: "Send a chat completion through the gateway. Supports the virtual 'duo' model (plan/execute/verify).",
      inputSchema: {
        type: "object",
        properties: {
          model: { type: "string", description: "Model id, e.g. duo, qwen3-4b, bonsai-27b" },
          messages: {
            type: "array",
            items: {
              type: "object",
              properties: { role: { type: "string", enum: ["system", "user", "assistant"] }, content: { type: "string" } },
              required: ["role", "content"],
            },
          },
          max_tokens: { type: "integer", description: "Default 512" },
          temperature: { type: "number", description: "Default 0.2" },
        },
        required: ["model", "messages"],
      },
    },
    {
      name: "benchmark",
      description: "Run the 6-task fast benchmark against one model and return a deterministic score plus per-task detail.",
      inputSchema: {
        type: "object",
        properties: { model: { type: "string" }, max_tokens: { type: "integer" } },
        required: ["model"],
      },
    },
    {
      name: "doctor",
      description: "Explain the current state: how many models are ready, unavailable or unsupported, and why.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  try {
    if (name === "gateway_status") {
      const health = await get("/healthz");
      const models = await get("/v1/models");
      const ready = models.data.filter((m) => m.status === "ready");
      const loaded = models.data.filter((m) => m.loaded);
      return text({
        gateway: BASE,
        health: health.status,
        models: models.data.length,
        ready: ready.map((m) => m.id),
        loaded: loaded.map((m) => m.id),
      });
    }
    if (name === "list_models") {
      const models = await get("/v1/models");
      return text(models.data);
    }
    if (name === "chat") {
      const data = await post("/v1/chat/completions", {
        model: args.model,
        messages: args.messages,
        max_tokens: args.max_tokens ?? 512,
        temperature: args.temperature ?? 0.2,
      });
      const msg = data.choices?.[0]?.message ?? {};
      return text({
        model: data.model ?? args.model,
        content: Array.isArray(msg.content) ? msg.content.map((b) => b.text ?? "").join(" ") : msg.content ?? "",
        tool_calls: msg.tool_calls ?? [],
        finish_reason: data.choices?.[0]?.finish_reason ?? null,
        usage: data.usage ?? null,
      });
    }
    if (name === "benchmark") {
      const rows = [];
      for (const [id, prompt, score] of BENCH) {
        const t0 = Date.now();
        let answer = "";
        let error = null;
        try {
          answer = await chatCompletion(args.model, prompt, args.max_tokens ?? 300);
        } catch (e) {
          error = String(e).slice(0, 200);
        }
        rows.push({ id, score: error ? 0 : score(answer), ms: Date.now() - t0, answer: answer.slice(0, 240), error });
      }
      const total = rows.reduce((a, r) => a + r.score, 0);
      return text({ model: args.model, score: `${total}/${rows.length}`, rows });
    }
    if (name === "doctor") {
      const models = await get("/v1/models");
      const by = (s) => models.data.filter((m) => m.status === s);
      return text({
        gateway: BASE,
        ready: by("ready").map((m) => m.id),
        unavailable: by("unavailable").map((m) => `${m.id}: ${m.reason ?? ""}`),
        unsupported: by("unsupported").map((m) => `${m.id}: ${m.reason ?? ""}`),
        hint: "unavailable models need their GGUF files next to the exe; unsupported ones are documented limitations.",
      });
    }
    throw new Error(`unknown tool ${name}`);
  } catch (e) {
    return fail(e);
  }
});

server.onerror = (e) => console.error("[localai-gateway-mcp]", e);
await server.connect(new StdioServerTransport());
