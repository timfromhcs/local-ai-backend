"""Precompute the OpenCode prompt prefix (system prompt + tool schemas incl. the ~38k-char skills list) ONCE, store the
llama-server slot state on SSD, and restore it at every server start.

The state is saved at the exact token boundary just before the volatile <env> block, so it is a true prefix of every
later OpenCode request (hybrid/recurrent models cannot roll a state back, so the cut must be exact).

Warmer(directory: Path, client: httpx.AsyncClient)
  Warmer.prefix_key(body) -> str
  filename(model_id, key) / known(model_id, key) -> bool
  await restore(base_url, model_id, key) -> bool ; await save(...) -> bool      (never raise; False on failure)
  await warm(base_url, model_id, prefix_body) -> WarmResult(state in {"restored","computed","failed"}, tokens, detail)
cut_point(pieces: list[bytes], text: bytes, marker=b"<env>", margin_bytes=64) -> int | None
"""
import json

import httpx

from localai_gateway.warmup import Warmer, cut_point

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
STABLE = "You are an agent.\n<available_skills>" + "skill " * 50 + "</available_skills>\n"


def body(system, tools=TOOLS, user="hi"):
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "tools": tools}


def env(cwd):
    return f"<env>\n  Working directory: {cwd}\n</env>\nExtra volatile tail."


# ---- key ---------------------------------------------------------------------------------------
def test_key_ignores_volatile_env_block_and_user_message():
    assert Warmer.prefix_key(body(STABLE + env("C:/a"), user="one")) == \
           Warmer.prefix_key(body(STABLE + env("C:/b/very/different"), user="another question"))


def test_key_changes_when_skills_or_tools_change():
    base = Warmer.prefix_key(body(STABLE + env("x")))
    assert Warmer.prefix_key(body(STABLE.replace("skill ", "skill2 ") + env("x"))) != base
    assert Warmer.prefix_key(body(STABLE + env("x"), tools=[])) != base


# ---- cut point ---------------------------------------------------------------------------------
def test_cut_point_stops_before_marker_with_margin():
    text = b"AAAA BBBB CCCC <env> DDDD"
    pieces = [b"AAAA", b" BBBB", b" CCCC", b" <env>", b" DDDD"]
    assert b"".join(pieces) == text
    assert cut_point(pieces, text, margin_bytes=0) == 3      # 3 pieces end exactly at the space before the marker region
    assert cut_point(pieces, text, margin_bytes=6) == 2      # margin keeps the cut away from the boundary token


def test_cut_point_returns_none_when_pieces_do_not_rebuild_text_or_marker_missing():
    assert cut_point([b"a", b"b"], b"abc") is None
    assert cut_point([b"ab", b"c"], b"abc") is None          # no marker


# ---- files / http ------------------------------------------------------------------------------
def test_filename_is_safe_and_known_reflects_files(tmp_path):
    w = Warmer(tmp_path, httpx.AsyncClient())
    name = w.filename("we/ird:model*", "abcdef0123456789abcdef")
    assert not set('/:*') & set(name) and name.endswith(".bin")
    assert not w.known("m", "k" * 20)
    (tmp_path / w.filename("m", "k" * 20)).write_bytes(b"x")
    assert w.known("m", "k" * 20)


async def test_save_and_restore_call_slot_endpoints(tmp_path):
    seen = []

    def handler(req):
        seen.append((req.url.path, req.url.params.get("action"), json.loads(req.content)))
        return httpx.Response(200, json={"id_slot": 0})

    w = Warmer(tmp_path, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await w.save("http://s", "m", "k" * 20) and await w.restore("http://s", "m", "k" * 20)
    fn = w.filename("m", "k" * 20)
    assert seen == [("/slots/0", "save", {"filename": fn}), ("/slots/0", "restore", {"filename": fn})]


async def test_http_failures_return_false_not_raise(tmp_path):
    def bad(req):
        return httpx.Response(404, json={"error": "no file"})

    def boom(req):
        raise httpx.ConnectError("down")

    for h in (bad, boom):
        w = Warmer(tmp_path, httpx.AsyncClient(transport=httpx.MockTransport(h)))
        assert await w.restore("http://x", "m", "k" * 20) is False and await w.save("http://x", "m", "k" * 20) is False


# ---- warm flow ---------------------------------------------------------------------------------
class FakeLlama:
    """apply-template -> tokenize(with_pieces) -> completion -> slots save/restore"""

    def __init__(self, files_ok=False):
        self.calls, self.files_ok = [], files_ok
        self.rendered = "SYS " + "skillword " * 30 + "<env> cwd </env> tail"

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path, payload = req.url.path, json.loads(req.content) if req.content else {}
        self.calls.append((path, req.url.params.get("action")))
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": self.rendered})
        if path == "/tokenize":
            words = self.rendered.split(" ")
            pieces = [(w if i == 0 else " " + w) for i, w in enumerate(words)]
            return httpx.Response(200, json={"tokens": [{"id": i, "piece": p} for i, p in enumerate(pieces)]})
        if path == "/completion":
            self.completion = payload
            return httpx.Response(200, json={"content": ""})
        if path == "/slots/0":
            if payload and req.url.params.get("action") == "restore" and not self.files_ok:
                return httpx.Response(404, json={"error": "missing"})
            return httpx.Response(200, json={"id_slot": 0})
        return httpx.Response(404)


async def test_warm_computes_once_then_restores_on_next_start(tmp_path):
    fake = FakeLlama()
    w = Warmer(tmp_path, httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    pb = body(STABLE + env("x"))
    r1 = await w.warm("http://s", "m", pb)
    assert r1.state == "computed" and r1.tokens > 5
    n_ids = len(fake.completion["prompt"])
    assert n_ids == r1.tokens and fake.completion["cache_prompt"] is True and fake.completion["n_predict"] == 0
    assert ("/slots/0", "save") in fake.calls and fake.rendered.count("skillword") > 0
    # the slot file now exists on "SSD" (fake server wrote nothing, so simulate llama-server's file):
    (tmp_path / w.filename("m", Warmer.prefix_key(pb))).write_bytes(b"slot")
    fake.files_ok, fake.calls = True, []
    r2 = await w.warm("http://s", "m", pb)
    assert r2.state == "restored" and ("/completion", None) not in fake.calls     # no recompute


async def test_warm_falls_back_to_compute_when_restore_fails(tmp_path):
    fake = FakeLlama(files_ok=False)
    w = Warmer(tmp_path, httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    pb = body(STABLE + env("x"))
    (tmp_path / w.filename("m", Warmer.prefix_key(pb))).write_bytes(b"corrupt")
    assert (await w.warm("http://s", "m", pb)).state == "computed"


async def test_warm_reports_failed_without_raising(tmp_path):
    def boom(req):
        raise httpx.ConnectError("down")

    w = Warmer(tmp_path, httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    r = await w.warm("http://s", "m", body(STABLE + env("x")))
    assert r.state == "failed" and r.detail
