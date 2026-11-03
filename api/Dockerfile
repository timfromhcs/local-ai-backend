# Gateway API only. No model weights, no llama.cpp runtimes.
# The container serves /healthz, /v1/models, /v1/chat/completions by proxying to a
# llama-server that runs outside the container (Vercel has no GPU; a PC/NAS does).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # the gateway refuses non-loopback binds unless this is set; inside a container 0.0.0.0 is
    # the only way to be reachable, and the port is published deliberately by the operator
    LOCALAI_ALLOW_PUBLIC=1 \
    LOCALAI_MAX_BODY_MB=8

WORKDIR /app

COPY pyproject.toml README.md models.yaml ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 gateway \
    && mkdir -p /app/cache && chown -R gateway:gateway /app

USER gateway
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3).status == 200 else 1)"

# 0.0.0.0 so the port is reachable from outside the container; the gateway itself
# still has no auth, so keep it on a private network or behind a proxy.
CMD ["python", "-m", "localai_gateway", "--host", "0.0.0.0", "--port", "8090"]
