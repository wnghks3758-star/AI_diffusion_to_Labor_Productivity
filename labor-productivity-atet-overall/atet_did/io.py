"""Atomic artifacts and content-addressed caches, without credentials."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_key(frame, settings):
    h = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    h.update(json.dumps(list(frame.columns)).encode())
    h.update(b"atet_did_v1")
    h.update(json.dumps(versions(), sort_keys=True).encode())
    h.update(source_hash().encode())
    return h.hexdigest()[:24]


def versions():
    try:
        importlib.metadata.version("xgboost-cpu")
        xgboost_package = "xgboost-cpu"
    except importlib.metadata.PackageNotFoundError:
        xgboost_package = "xgboost"
    return {
        "python": platform.python_version(),
        **{
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                xgboost_package,
                "joblib",
            ]
        },
    }


def source_hash():
    """Fingerprint the entire standalone runtime, including CLI and sample rules."""
    h = hashlib.sha256()
    package = Path(__file__).resolve().parent
    files = sorted(package.glob("*.py")) + [package.parent / "run_atet.py"]
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()
