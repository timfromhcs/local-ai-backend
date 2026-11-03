# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a ONE-FILE localai-gateway.exe.

Deliberately excludes model weights (GB) and the llama/sd runtimes (bin/):
the exe contains only the gateway + its Python dependencies. Point it at
models.yaml and bin/ next to the exe (or set LOCALAI_HOME).
"""
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("fastapi")
    + collect_submodules("pydantic")
    + [
        "yaml",
        "psutil",
        "httpx",
        "anyio",
        "h11",
        "click",
        "uvloop",
    ]
)

a = Analysis(
    ["entry.py"],
    pathex=["../src"],
    binaries=[],
    datas=[("../models.yaml", ".")],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "onnxruntime",
        "onnx",
        "transformers",
        "numpy",
        "tkinter",
        "matplotlib",
        "pytest",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="localai-gateway",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
