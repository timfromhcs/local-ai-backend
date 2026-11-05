"""The runtime launcher has vanished twice on this machine; the gateway must heal itself.

Observed: `bin/llama/llama-server.exe` disappears while its 46 sibling binaries stay, no antivirus
event, no git action (bin/ is ignored). The result was a 90 s start timeout and a 503 that mentioned
a tokenizer warning instead of the missing file. The fix is not a theory about the cause - it is that a
shipped zip is enough to restore the launcher.
"""
import zipfile
from pathlib import Path

from localai_gateway.runtime import default_zip, ensure_runtime, restore_from_zip


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_missing_launcher_is_restored_from_the_zip(tmp_path):
    archive = make_zip(tmp_path / "llama-x-vulkan.zip", {
        "llama-server.exe": b"MZfake",
        "llama-server-impl.dll": b"DLL",
        "ggml-base.dll": b"DLL",
    })
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"

    path, note = ensure_runtime(exe, archive)

    assert path == exe and exe.exists()
    assert exe.read_bytes() == b"MZfake"
    assert (exe.parent / "llama-server-impl.dll").exists(), "the launcher needs its siblings"
    assert "restored" in note


def test_an_existing_launcher_is_never_overwritten(tmp_path):
    archive = make_zip(tmp_path / "llama-x-vulkan.zip", {"llama-server.exe": b"NEW"})
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"OLD")

    path, note = ensure_runtime(exe, archive)

    assert path == exe and note == ""
    assert exe.read_bytes() == b"OLD", "a working runtime must not be replaced"


def test_without_a_zip_the_note_says_what_is_missing(tmp_path):
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"
    path, note = ensure_runtime(exe, None)
    assert path == exe and not exe.exists()
    assert "missing" in note and "no runtime zip" in note


def test_a_zip_without_the_launcher_is_reported(tmp_path):
    archive = make_zip(tmp_path / "llama-x-vulkan.zip", {"README.txt": b"nothing here"})
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"
    _, note = ensure_runtime(exe, archive)
    assert "does not contain" in note and not exe.exists()


def test_a_corrupt_zip_does_not_raise(tmp_path):
    bad = tmp_path / "llama-bad.zip"
    bad.write_bytes(b"not a zip")
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"
    _, note = ensure_runtime(exe, bad)
    assert "could not restore" in note


def test_default_zip_finds_the_shipped_archive(tmp_path):
    make_zip(tmp_path / "llama-prism-b12128-d7b2896-bin-win-vulkan-x64.zip", {"llama-server.exe": b"x"})
    (tmp_path / "other.zip").write_bytes(b"")
    found = default_zip(tmp_path)
    assert found is not None and found.name.startswith("llama-prism")


def test_default_zip_is_none_when_nothing_is_shipped(tmp_path):
    assert default_zip(tmp_path) is None


def test_restore_keeps_a_flat_archive_layout(tmp_path):
    archive = make_zip(tmp_path / "llama-x.zip", {
        "llama-server.exe": b"a",
        "subdir/ignored.txt": b"b",
    })
    exe = tmp_path / "bin" / "llama" / "llama-server.exe"
    restore_from_zip(archive, exe)
    assert exe.exists()
    assert not (exe.parent / "ignored.txt").exists(), "nested paths are not flattened into bin/"
