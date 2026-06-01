"""Common helpers shared by all capture/*.py scripts."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def get_environment_metadata() -> dict[str, Any]:
    """Capture hostname / OS / Python / CUDA / cuQuantum / GPU info."""
    meta: dict[str, Any] = {
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import cuquantum  # noqa: F401
        meta["cuquantum"] = getattr(cuquantum, "__version__", "unknown")
    except Exception as e:  # pragma: no cover
        meta["cuquantum"] = f"unavailable: {e!r}"
    try:
        import cupy as cp
        meta["cupy"] = cp.__version__
        meta["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        meta["gpu_name"] = name
        meta["gpu_id"] = int(dev.id)
        meta["gpu_compute_cap"] = f"{props['major']}.{props['minor']}"
    except Exception as e:  # pragma: no cover
        meta["cupy"] = f"unavailable: {e!r}"
    return meta


def stable_hash(*objs: Any) -> str:
    """Deterministic short hash for use as a parameter id suffix."""
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, np.ndarray):
            h.update(o.tobytes())
            h.update(str(o.shape).encode())
            h.update(str(o.dtype).encode())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:8]


@dataclass
class OracleCase:
    """One capture record."""
    api: str           # matches verification.metrics.METRICS keys
    param_id: str      # filename-safe id; must be unique within (api,)
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def write_case(out_root: Path, library: str, case: OracleCase,
               env_metadata: dict[str, Any]) -> Path:
    """Persist an OracleCase as a single .npz file plus a json sidecar.

    Layout: ``<out_root>/<library>/<api>/<param_id>.npz``.
    """
    api_dir = case.api.split(".", 1)[1] if "." in case.api else case.api
    target_dir = out_root / library / api_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_path = target_dir / f"{case.param_id}.npz"
    json_path = target_dir / f"{case.param_id}.json"

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}

    def _bucket(prefix: str, d: dict[str, Any]) -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}"
            if isinstance(v, np.ndarray):
                arrays[key] = v
            else:
                scalars[key] = v

    _bucket("input", case.inputs)
    _bucket("output", case.outputs)
    np.savez_compressed(npz_path, **arrays)

    sidecar = {
        "api": case.api,
        "param_id": case.param_id,
        "params": case.params,
        "metadata": case.metadata,
        "scalars": scalars,
        "env": env_metadata,
        "array_keys": sorted(arrays.keys()),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))
    return npz_path


def load_case(npz_path: Path) -> dict[str, Any]:
    """Inverse of write_case. Returns dict with input/output/params/metadata."""
    json_path = npz_path.with_suffix(".json")
    sidecar = json.loads(json_path.read_text())
    arrays = dict(np.load(npz_path, allow_pickle=False))
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for k, v in arrays.items():
        if k.startswith("input."):
            inputs[k[len("input."):]] = v
        elif k.startswith("output."):
            outputs[k[len("output."):]] = v
    for k, v in sidecar.get("scalars", {}).items():
        if k.startswith("input."):
            inputs[k[len("input."):]] = v
        elif k.startswith("output."):
            outputs[k[len("output."):]] = v
    return {
        "api": sidecar["api"],
        "param_id": sidecar["param_id"],
        "params": sidecar.get("params", {}),
        "metadata": sidecar.get("metadata", {}),
        "env": sidecar.get("env", {}),
        "inputs": inputs,
        "outputs": outputs,
    }


def common_argparser(library: str, default_seed: int = 0) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Capture cuQuantum {library} oracle data")
    p.add_argument("--out", type=Path, required=True,
                   help="oracle root, e.g. verification/oracle/v1.0.0")
    p.add_argument("--seed", type=int, default=default_seed)
    p.add_argument("--small", action="store_true",
                   help="capture only the smallest cases (CI smoke).")
    p.add_argument("--filter", default=None,
                   help="substring filter on param_id; only matching cases captured.")
    p.add_argument("--dry-run", action="store_true")
    return p


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_unitary(rng: np.random.Generator, dim: int, dtype: str) -> np.ndarray:
    """Haar-random unitary via QR of a complex Ginibre matrix."""
    real = rng.standard_normal((dim, dim))
    imag = rng.standard_normal((dim, dim))
    a = (real + 1j * imag).astype(dtype)
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph[np.newaxis, :]
    return q.astype(dtype)


def random_statevector(rng: np.random.Generator, n_qubits: int, dtype: str) -> np.ndarray:
    dim = 1 << n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    psi = (real + 1j * imag).astype(dtype)
    psi /= np.linalg.norm(psi)
    return psi
