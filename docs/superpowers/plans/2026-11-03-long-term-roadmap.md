# Long-term plan: LocalAI Gateway 2026/2027

Based on the repository state at v0.4.0 and SOTA research on supply chain, evaluation and
MCP. Rule: each phase ends with a gate that runs in CI; no phase is "done" because it was
written down.

## Where the project stands

| Area | State | Evidence |
|---|---|---|
| Gateway API | frozen, 12 paths, snapshot gate | `docs/openapi.json`, `tests/test_api_contract.py` |
| Model runtime | llama.cpp Vulkan, 27B/4B/1.5B measured | `docs/perf-log.md` |
| Planning | SAT placement + device autotune | `tests/test_placement_autotune.py` |
| Browser | ONNX + GGUF/WASM, 3 languages, a11y gate | production page, `audit-web.mjs` |
| Containers | API image + GHCR | `ghcr.io/...:0.4.0` |
| MCP | 16 servers, 2 of them ours | probes in `~/.config/opencode/scripts` |
| RSI | capture + dataset + CPU LoRA path, one smoke run | `rsi/runs/20261103-063851` |
| Release | v0.4.0, EXE + wheel + sdist + GHCR, ruleset | tag, release page |
| **Gaps** | no SBOM/provenance, no dependency audit, no golden eval gate, npm/PyPI unpublished | this plan |

## SOTA notes that shape the plan

1. **Artifact attestations** are the 2026 default for binaries: `actions/attest-build-provenance`
   for the "how/where", `actions/attest` with a CycloneDX `sbom-path` for the "what". Verification
   runs through `gh attestation verify`, so a user can prove the EXE came from this repo's workflow.
2. **Golden datasets** work best as a fast PR gate of tens of items, versioned, with a stored
   baseline and a tolerance; full regression sets (hundreds to thousands) run nightly. The gate
   fails on regression, not on absolute quality, which is the only way it survives model noise.
3. **MCP** servers should be thin, deterministic clients of a documented API; our gateway and
   rsi-capture servers already follow that, and the API snapshot is the contract they rely on.

## Phases

### Phase A — supply chain and evaluation (this cycle)
- `scripts/sbom.py`: CycloneDX 1.5 from the installed distributions, written to `dist/sbom.cdx.json`.
- Release: attach the SBOM, attest the EXE and wheel (separate `attest.yml` so a broken action
  version cannot block a release).
- CI: `scripts/scan_secrets.py` (own implementation, no third-party action) and `pip-audit`
  (fail on HIGH/CRITICAL).
- `.github/dependabot.yml` for pip, npm and actions.
- Golden eval gate: `bench/golden.json` + `tests/test_golden_eval.py`; the real scorers run
  against committed answers and must match the stored score exactly, aggregate must equal the
  baseline. Catches scorer drift without a model.
- Gate: all jobs green, SBOM valid CycloneDX, `gh attestation verify` documented.

**Shipped (PR #16):** SBOM job in the release workflow (43 components, version read from the
source tree so an editable install cannot lie), `attest.yml` as its own workflow with
`id-token: write` + `attestations: write`, a `security` CI job (secret scan + pip-audit + SBOM
smoke), Dependabot for pip/npm/actions, and the golden eval gate (16 pinned cases; every scorer
has a passing and a failing case and the aggregate is pinned). The secret scanner immediately
caught a fake token in a test fixture; fixtures now carry an explicit `secret-scan: allow` marker.

**Dependabot triage (PR #24):** five updates applied in one reviewed change (checkout 6->7,
download-artifact 6->8, setup-buildx 3->4, pydantic >=2.14.2, setuptools >=84.2.1). The five
individual PRs were closed with that reason: sequential merges kept landing behind `main` because
of the required up-to-date branches, and batching is one CI run instead of five. One PR was
rejected outright: the Playwright bump to `1.65.0-alpha-1792021904000` is a timestamp-suffixed
alpha, not a published release, and our pin must stay a version that exists.

### Phase B — dogfooding the RSI loop (next)
- Run every autonomous cycle through `rsi-capture` (start → tool → verification → finalize).
- Reach 20+ verified pairs, then a real LoRA run on the 1.5B router in `.venv-train`
  (checkpoints, hours) and compare against the fast bench with `rsi_lab.train.promote`.
- Gate: a promotion decision with numbers, or an explicit "candidate lost, champion kept".

**Progress:** the capture loop is wired into the hourly wake prompt, so every cycle records
itself. Two verified pairs exist (392 + 124 tokens) from real work: the supply-chain cycle and
the placement-route verification. The dataset builder now deduplicates re-exported pairs so one
episode cannot be trained twice. 18 more verified sessions are needed before a 1.5B run is worth
the CPU hours; until then `--plan` is the honest output.

### Phase C — distribution and clients (needs credentials)
- Publish the wheel (PyPI Trusted Publishing) and the npm launcher; both workflows exist and
  skip honestly today.
- Generate a typed client from `docs/openapi.json` so third parties do not hand-roll requests.
- Gate: `pip install localai-gateway` and `npx @hcsmedia/localai-gateway --doctor` work on a
  clean machine.

### Phase D — 2027 runtime work (research, not promised)
- Speculative decoding with the 0.5B drafter and a measured break-even check (the earlier
  measurement says verify cost is not free on this iGPU: only keep it above 46 % acceptance).
- OPFS + SQLite-WASM for durable per-user data beyond IndexedDB size limits.
- PWA install for the phone path, with an offline shell and explicit model download sizes.
- Gate: each item ships with a measured number or is deleted from the plan.

## Non-goals (stated so they stop coming back)

- 50-100 tok/s on a 27B model on 88 GB/s of shared memory.
- Training a frontier model from scratch on this iGPU.
- Running the 27B model in a browser or on Vercel's container runtime.

## How the loop runs

`OpenCodeHourlyWake` posts a prompt into the session every hour; each cycle takes the next open
item, runs the gates, commits through a PR and updates this file. Two consecutive green release
runs remain the bar for calling anything stable.
