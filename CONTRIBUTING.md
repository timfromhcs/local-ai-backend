# Contributing

This repository is built by a loop: one change, its gate, a pull request through the ruleset.
The rules are deliberately boring.

## Ground rules

1. **No model weights, no binaries, no caches in git.** `*.gguf`, `*.onnx`, `*.safetensors`,
   `bin/`, `dist/`, `build/`, `cache/`, `rsi/` are ignored. The EXE ships without weights on
   purpose.
2. **No unmeasured claims.** Every performance statement lives next to a command or a log in
   `docs/perf-log.md`. "Should be faster" is not a change.
3. **The gates are the contract.** A PR is mergeable when the ruleset checks pass:
   `tests (ubuntu-latest)`, `tests (windows-latest)`, `web static gate`, plus the informational
   jobs (`accessibility audit`, `secret scan + dependency audit`, `one-file exe`, `container`).
4. **One concern per PR.** Bug fix and refactor in separate PRs; a failing test and its fix may
   share one.

## Local gates (run these before pushing)

```powershell
.\.venv\Scripts\python -m pytest -m "not real" -q     # 272+ tests, no model downloads
node scripts\check-web.mjs                            # vercel.json, CSP, i18n parity, SEO, assets
node scripts\audit-web.mjs                            # Chromium: contrast, tap targets, focus, motion, console errors
node scripts\verify-storage.mjs                       # IndexedDB + OPFS round-trip, history rendering
node scripts\verify-pwa-offline.mjs                   # service worker serves the shell offline
python scripts\scan_secrets.py                        # no tokens in tracked files
python scripts\sbom.py --out dist\sbom.cdx.json       # CycloneDX 1.5
python -m localai_gateway --self-test                 # API works in-process, no port
python scripts\release-check.py                       # all 12 gates with one verdict
```

Deploying the demo: use `scripts\deploy-web.ps1`. It runs the web gate, stamps the service-worker
cache version (so the first load after a deploy is not stale), deploys, then verifies production.
Plain `vercel deploy` still works but can serve the previous `app.js` once.

`npm run` is not used: the web demo has no build step, on purpose.

## Commit and PR shape

- Commits: `feat(area):`, `fix(area):`, `docs:`, `test:`, `chore(deps):` — one sentence that says
  what changed and why, then a short body with the evidence (numbers, commands, test counts).
- PRs: title mirrors the commit; description states the gate output and what remains open.
- Branches are merged with squash; `main` is protected, so a direct push is rejected by design.

## Adding a model

1. Put the GGUF next to the binary (never in git).
2. Add an entry to `models.yaml` with `est_mem_mb`, the real flags and a note.
3. Run `localai-gateway --doctor` and `GET /v1/placement`; if the model does not fit, say so in
   the entry instead of pretending.
4. Unsupported models stay listed with their loader error, not deleted.

## Adding a route

1. Add the route and a test in `tests/test_api_contract.py` (stable paths) or a new file.
2. Regenerate the snapshot: `python scripts/snapshot-openapi.py`.
3. If a client or MCP server consumes it, update that in the same PR.

## Reporting a security issue

See [SECURITY.md](SECURITY.md). Do not open a public issue for a vulnerability.

## Licence

MIT. By contributing you agree your changes are licensed under the same terms. Model weights keep
their own licences and are never redistributed here.
