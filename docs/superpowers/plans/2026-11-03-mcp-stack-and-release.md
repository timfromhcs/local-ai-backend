# Plan: MCP stack hardened → single EXE → WebGPU demo → release

Status 2026-11-03. Supersedes the MCP/infra parts of `2026-11-03-autoloop-exe-web-release.md`.
Rule: only measured facts, no claim without a command that proves it.

## 0. Where the previous agent stopped (verified)

| Milestone | State | Evidence |
|---|---|---|
| M1 mode controller (fast/duo/bigout) | done | `src/localai_gateway/modes.py`, `tests/test_modes.py` |
| M2 RSI core (scrub/harvest/clean/split/gate) | done | last commit `e004db0`, `tests/rsi/` |
| M2 loop state machine + dry run | **open** | no `rsi/state.json`, no loop entry point |
| M3 Docker SWE runner | **open** | no runner code |
| M4 LoRA train/merge/bench | **open** | `training/gateway_sft.jsonl` (1 file, 1.5 KB) untracked |
| M5 solo EXE | **open** | no spec file, no build |
| M6 web demo | **open** | no `web/` directory |
| M7 release | **blocked** | no GitHub remote, no README, no CI |

Tests at the stop point: `117 passed in 45.37s` (`pytest -m "not real"`).

## 1. Why the MCPs were broken (root causes, all fixed and verified)

| Symptom | Cause | Fix |
|---|---|---|
| github, playwright, ast-grep, fetch, memory, sequential-thinking, blender, jev all "failed" in this project | `opencode.jsonc` in the repo redefined every server as `{"command": ["true"], "enabled": false}`. OpenCode still spawned them; `true` is not a Windows command → `exit code 1` for all 8. | Project config no longer redefines `mcp`; the global config applies. |
| Vercel MCP 401 → "requires a login before it can register a client" | Remote MCP needs OAuth; no token was registered for `https://mcp.vercel.com`. | OAuth completed in OpenCode; `vercel` now connects with 241 tools. |
| better-design 401 | Same OAuth requirement, transient. | Connected via the existing bearer token; 45 tools. |
| Servers die on reboot / after cache clean | `npx -y` and `uvx` resolve and download at *startup*; PATH, registry or cache loss breaks them. | Pinned installs in `.config/opencode/mcp`, absolute script paths, uv-tool executables, browser pre-installed. |
| New `local-crawl` MCP timed out (`-32001`) although a raw probe worked | (a) hand-rolled JSON-RPC loop instead of the official SDK; (b) a `"timeout": 120000` field on the local entry made OpenCode abort the connection in some directories. | Server rewritten on `@modelcontextprotocol/sdk`; `timeout` removed. `opencode mcp list` → `✓ local-crawl`. |
| `chroma-mcp` crashed on import | pydantic 2.12 removed `eval_type_backport`, which the Python MCP SDK imports. | `uv tool install --force --with "pydantic<2.12" chroma-mcp`. |

### Always-on guarantee (honest scope)
MCP servers are stdio children of OpenCode, so they cannot be Windows services. What is guaranteed now:

* `scripts/mcp-bootstrap.ps1` — idempotently reinstalls npm packages, the uv tools, Playwright Chromium and checks fixed binaries.
* `scripts/mcp-health.ps1` — real `initialize` + `tools/list` probe for every server (stdio) and reachability probe (http), writes `mcp-status.json`, `-Repair` runs the bootstrap and retries.
* `scripts/jev-guard.ps1` — verifies a **real completion** from the System-1 llama-server on `:8080` (the key-less `/health` check hid a 401) and restarts it with the measured flags if needed.
* Scheduled tasks `OpenCodeMCPHealth` (logon + startup +45 s + every 4 h) and `OpenCodeJevGuard` (logon + every 10 min), registered by `scripts/register-mcp-autostart.ps1`.

Result now: 14/14 servers reachable, jev answering completions, knowledge base seeded.

## 2. MCP fleet (verified tool counts)

| Server | Tools | Purpose |
|---|---|---|
| github | 91 | repo, PR, issue, actions |
| playwright | 25 | E2E, DOM inspection |
| chrome-devtools | 32 | fast CDP browser, traces, console |
| local-crawl | 6 | local crawler (HTTP fast path → local Chromium, disk cache, robots.txt) |
| knowledge-db (Chroma) | 13 | local vector DB, 20 seeded best-practice docs |
| basic-memory | 12 | durable markdown knowledge base |
| memory | 9 | knowledge graph |
| sequential-thinking | 1 | structured reasoning |
| fetch | 1 | single-page fetch |
| ast-grep | 4 | structural search |
| blender | 9 | 3D (needs Blender running) |
| context7 | 2 | library docs |
| better-design | 45 | design systems, UI review |
| vercel | 241 | deployments, domains, logs |
| system-one-jev | 21 | fast decisions, routing, local status |

## 3. Milestones from here

**M5 Solo EXE** — PyInstaller one-file `localai-gateway.exe`: no model files, no `bin/`, runtime path resolution via `sys._MEIPASS`, `doctor` output, smoke test that starts the exe and calls `/healthz` + `/v1/models`.

**M6 Web demo** — static site in `web/`, deployed to Vercel: WebGPU LLM in the browser (`@mlc-ai/web-llm`), hardware gating (adapter, limits, memory → model tier), "connect to your gateway" mode, EN/DE/FR, honest "zero to hero" guide, logo + illustrations. Verified with Playwright at 390 px and 1440 px, then preview → production.

**M7 Release** — secret scan, README with measured numbers and real limitations, CI (pytest + exe build), GitHub repo, tag, release artifact (exe + wheel, no models).

## 4. RSI Lab v3 — chat → training data → text-diffusion Jev (requested, staged)

Realistic staging, because "train a text diffusion model from scratch on a 12 CU RDNA2 iGPU" cannot happen in one pass:

1. **Capture (buildable now)** — `rsi-mcp` server: append-only JSONL of every session event, derived training pairs with **only executed/verified intermediate steps**, synthetic steps marked `synthetic:true` and never claimed as reasoning. Tools: `start_capture`, `log_event`, `finalize_session`, `build_pairs`, `dataset_stats`.
2. **Distill (buildable now)** — the pairs feed the existing scrub/clean/split chain; LoRA on the 1.5B Jev model on CPU; promotion gate on the fast bench. Jev stays a classifier/router first.
3. **Diffusion research (staged)** — start with a tiny byte-level / token-level diffusion LM (≤30 M params, 4 k steps) on this GPU to measure throughput, then scale only if the measured tokens/s justifies it. Benchmarks: fast suite (tool-call validity, routing accuracy, 20 HumanEval+ items, ~5 min) and long suite (full eval + regression, hours). No promise of "general reasoning" before the small model beats the classifier on the fast suite.

Honest limit: Vulkan on a Radeon 680M has no matrix cores; measured ~1.7 TFLOP/s effective. A from-scratch diffusion model here will be a research vehicle, not a frontier model.

## 5. Release gate
`pytest -m "not real"` green · exe smoke tested · web demo verified in a real browser · secret scan clean · `.gitignore` covers models/bin/caches/training artifacts · README states only measured facts · then GitHub + Vercel, each once, in order.
