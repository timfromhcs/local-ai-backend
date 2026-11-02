"""Inspect .onnx files with onnxruntime (CPU) and report IO signatures + hashes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _static_io(path: Path) -> tuple[list, list]:
    """IO signature straight from the protobuf graph (no weights needed)."""
    import onnx
    from onnx import TensorProto

    m = onnx.load(str(path), load_external_data=False)
    init = {i.name for i in m.graph.initializer}

    def fmt(vs):
        out = []
        for v in vs:
            if v.name in init:
                continue
            t = v.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None) for d in t.shape.dim]
            out.append({"name": v.name, "shape": dims, "dtype": TensorProto.DataType.Name(t.elem_type)})
        return out

    return fmt(m.graph.input), fmt(m.graph.output)


def inspect_onnx(path: Path) -> dict:
    import onnxruntime as ort

    path = Path(path)
    info: dict = {"file": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:  # record the real error, never fake
        msg = str(e).strip()
        info["error"] = msg.splitlines()[0] if msg else repr(e)
        info["inputs"], info["outputs"] = _static_io(path)
        info["loadable"] = False
        return info
    info["loadable"] = True

    def fmt(xs):
        return [{"name": x.name, "shape": list(x.shape), "dtype": x.type} for x in xs]

    info["inputs"] = fmt(sess.get_inputs())
    info["outputs"] = fmt(sess.get_outputs())
    return info


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(inspect_onnx(Path(p)), indent=2))
