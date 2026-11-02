from rsi_lab.scrub import scrub_text, contains_secret

CANARIES = [
    "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",       # GitHub PAT
    "github_pat_" + "11ABCDEFG0abcdefghijkl_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvw",
    "bd_live_" + "F0BCFou3zVWJ7YvNtf2nkgdhdBldi9sm",        # better-design style key
    "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD",
    "AKIA" + "ABCDEFGHIJKLMNOP",                              # AWS access key id
    "xoxb-" + "123456789012-1234567890123-abcdefghijklmnopqrstuvwx",
    "hf_" + "abcdefghijklmnopqrstuvwxyzABCDEFGH",
]


def test_each_canary_is_redacted():
    for c in CANARIES:
        out = scrub_text(f"token is {c} ok")
        assert c not in out and "[REDACTED" in out, c
        assert not contains_secret(out)


def test_bearer_and_env_assignments_redacted():
    out = scrub_text('Authorization: Bearer abc.def-ghi_123456789\nGITHUB_TOKEN=supersecretvalue123\nPASSWORD="hunter2hunter2"')
    assert "abc.def-ghi_123456789" not in out and "supersecretvalue123" not in out and "hunter2hunter2" not in out


def test_private_key_block_and_email_redacted():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n-----END RSA PRIVATE KEY-----"
    out = scrub_text(f"key:\n{pem}\nmail me at person.name@example.com")
    assert "MIIEabc" not in out and "person.name@example.com" not in out


def test_normal_text_untouched():
    s = "def hello():\n    return 'hi'  # port 18090, sha 0f86314, version 0.7.0"
    assert scrub_text(s) == s and not contains_secret(s)


def test_idempotent():
    s = scrub_text("x " + CANARIES[0])
    assert scrub_text(s) == s
