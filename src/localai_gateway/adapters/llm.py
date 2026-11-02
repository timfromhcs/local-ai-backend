"""OpenAI chat-completions passthrough to a llama-server endpoint (non-stream and raw SSE stream)."""
from __future__ import annotations

from typing import AsyncIterator

import httpx


class BackendError(Exception):
    """The backend answered with an error, or could not be reached."""


async def forward_chat(base_url: str, body: dict, client: httpx.AsyncClient) -> dict:
    try:
        r = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body)
    except httpx.HTTPError as exc:
        raise BackendError(f"backend unreachable: {exc!r}") from exc
    if r.status_code >= 400:
        raise BackendError(f"backend HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as exc:
        raise BackendError("backend returned non-JSON") from exc


async def open_stream(base_url: str, body: dict, client: httpx.AsyncClient) -> httpx.Response:
    """Send the request and return the streaming response ONLY if the backend accepted it (status < 400)."""
    req = client.build_request("POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise BackendError(f"backend unreachable: {exc!r}") from exc
    if resp.status_code >= 400:
        text = (await resp.aread()).decode("utf8", "replace")[:200]
        await resp.aclose()
        raise BackendError(f"backend HTTP {resp.status_code}: {text}")
    return resp


async def iter_stream(resp: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()
