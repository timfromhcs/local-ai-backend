"""Precompute the OpenCode prompt prefix (system prompt + tool schemas incl. the ~38k-char skills list) ONCE, store the
llama-server slot state on SSD, and restore it at every server start.

The state is saved at the exact token boundary just before the volatile <env> block, so it is a true prefix of every
later OpenCode request (hybrid/recurrent models cannot roll a state back, so the cut must be exact).

Warmer(directory: Path, client: httpx.AsyncClient)
  Warmer.prefix_key(body) -> str
  filename(model_id, key) / known(model_id, key) -> bool
  await restore(base_url, model_id, key) -> bool ; await save(...) -> bool      (never raise; False on failure)
  await warm(base_url, model_id, prefix_body) -> WarmResult(state in {"restored","computed","failed"}, tokens, detail)
cut_point(pieces: list[bytes], text: bytes, limit_bytes: int) -> int | None   (tokens that end within limit_bytes)
normalize_for_cache(body) -> body   moves the volatile <env>..</env> block from the system prompt to the first user message
"""
import json

import httpx

from localai_gateway.warmup import Warmer, cut_point, normalize_for_cache

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


# ---- cut point / normalize ----------------------------------------------------------------------
def test_cut_point_counts_tokens_ending_within_limit():
    text = b"AAAA BBBB CCCC DDDD"
    pieces = [b"AAAA", b" BBBB", b" CCCC", b" DDDD"]
    assert cut_point(pieces, text, 14) == 3 and cut_point(pieces, text, 13) == 2 and cut_point(pieces, text, 3) == 0


def test_cut_point_none_when_pieces_do_not_rebuild_text():
    assert cut_point([b"a", b"b"], b"abc", 3) is None


def test_normalize_moves_env_block_into_first_user_message():
    b = body(STABLE + env("C:/a"), user="question")
    n = normalize_for_cache(b)
    sys_text, user_text = n["messages"][0]["content"], n["messages"][1]["content"]
    assert "<env>" not in sys_text and sys_text.startswith("You are an agent.")
    assert user_text.startswith("<env>") and "C:/a" in user_text and user_text.endswith("question")
    assert b["messages"][0]["content"].count("<env>") == 1                     # input not mutated
    assert normalize_for_cache(n) == n                                          # idempotent


def test_normalize_leaves_bodies_without_env_untouched_and_handles_list_content():
    assert normalize_for_cache(body("plain")) == body("plain")
    b = {"messages": [{"role": "system", "content": [{"type": "text", "text": "S " + env("x")}]},
                      {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    n = normalize_for_cache(b)
    assert "<env>" not in json.dumps(n["messages"][0]) and "<env>" in json.dumps(n["messages"][1])


def test_normalized_keys_equal_across_different_cwds():
    a = Warmer.prefix_key(normalize_for_cache(body(STABLE + env("C:/a"), user="one")))
    b = Warmer.prefix_key(normalize_for_cache(body(STABLE + env("C:/zzz"), user="two")))
    assert a == b


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
        self.stable = "SYS " + "skillword " * 30

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path, payload = req.url.path, json.loads(req.content) if req.content else {}
        self.calls.append((path, req.url.params.get("action")))
        if path == "/apply-template":
            user = [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]
            self.rendered = self.stable + "\n<user>" + (user if isinstance(user, str) else "x")
            return httpx.Response(200, json={"prompt": self.rendered})
        if path == "/tokenize":
            self.rendered = payload["content"]
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
    assert ("/slots/0", "save") in fake.calls
    assert fake.calls.count(("/apply-template", None)) == 2                     # two renderings -> template-agnostic boundary
    assert n_ids <= len(fake.stable.split(" "))                                  # cut lies inside the stable part
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
