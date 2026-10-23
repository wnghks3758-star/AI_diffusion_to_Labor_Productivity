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
        + "\n"
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
    h.update(b"labor_did_v1")
    h.update(json.dumps(versions(), sort_keys=True).encode())
    # Invalidate numerical caches after an implementation change as well as a package change.
    for name in [
        "models.py",
        "gps.py",
        "support.py",
        "weights.py",
        "data.py",
        "estimation.py",
        "aggregation.py",
        "score.py",
    ]:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:24]


def versions():
    return {
        "python": platform.python_version(),
        **{
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost-cpu",
                "joblib",
            ]
        },
    }
