# Security policy

## Scope

This project runs a **local** gateway on `127.0.0.1` and a **static** demo page. It ships no
server-side service, no account system and no telemetry.

In scope:

- the gateway API (`src/localai_gateway/`): request handling, path/traversal issues in model
  loading, secret handling in logs and captures, denial of service through unbounded requests
- the RSI capture pipeline (`src/rsi_lab/`): secret leakage into captured files, verification
  bypass that lets unverified work become training data
- the web demo (`web/`): XSS, CSP bypasses, data exfiltration of the local-first store
- the release artifacts: SBOM mismatch, checksum mismatch, tampered binaries

Out of scope, because it is the design:

- **No authentication.** The gateway is meant to be reachable only from the local machine. It
  **refuses to bind a non-loopback address** unless `LOCALAI_ALLOW_PUBLIC=1` is set, and rejects
  request bodies above `LOCALAI_MAX_BODY_MB` (default 8 MB, `413 payload_too_large`). If you expose
  it anyway, put an authenticating proxy in front of it; that is your responsibility, not a bug.
- **Model output quality.** A 4B model giving a wrong answer is a limitation, not a vulnerability.
- **The unsigned EXE.** Windows SmartScreen warns by design; a code-signing certificate is not
  configured yet and is documented in the README.
- **Denial of service by asking for a 27B model on a small machine.** The scheduler budgets RAM
  and returns `does_not_fit`, but it does not protect against someone with local shell access.

## Reporting

Use GitHub's private vulnerability reporting:
**Security → Advisories → Report a vulnerability** in
`github.com/timfromhcs/local-ai-backend`. Please do not open a public issue first.

Include: affected version (`localai-gateway --version`), platform, a minimal reproduction, and
what you expected to happen. If you have a fix, a patch in the advisory is welcome.

## What to expect

- Acknowledgement within 7 days.
- A fix or an explicit "not a vulnerability, here is why" within 30 days for in-scope reports.
- Credit in the release notes if you want it.

## Verifying a release

```powershell
gh release download <tag> -p localai-gateway.exe -p SHA256SUMS.txt -p sbom.cdx.json
Get-FileHash .\localai-gateway.exe -Algorithm SHA256   # must match SHA256SUMS.txt
gh attestation verify .\localai-gateway.exe --repo timfromhcs/local-ai-backend
```

The SBOM lists every shipped dependency; CI regenerates and validates it on every release.
