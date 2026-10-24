"""Inspect runtime and synthetic XGBoost sampling reproducibility without uploading firm data."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb


def runtime_info():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "xgboost-cpu", "joblib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(os=platform.system(), machine=platform.machine(), python=sys.version.split()[0],
                xgboost_runtime=xgb.__version__, packages=packages, build=xgb.build_info())


def sampling_fingerprints():
    rng = np.random.default_rng(1738)
    X = rng.integers(-2000, 2000, size=(128, 9)).astype(np.float32) / 1024
    y = 2*X[:, 0] + .25*X[:, 1]**2 - X[:, 2] + .125*X[:, 4]*X[:, 5]
    rows = {}
    for name, subsample, columns in [("no_sampling", 1., 1.), ("rows_only", .7, 1.), ("columns_only", 1., .7), ("both", .7, .7)]:
        model = xgb.XGBRegressor(n_estimators=30, max_depth=3, min_child_weight=5, learning_rate=.05,
                                 tree_method="exact", objective="reg:squarederror", subsample=subsample,
                                 colsample_bytree=columns, reg_lambda=3., random_state=2026102409, n_jobs=1)
        model.fit(X, y)
        prediction = model.predict(X).astype("<f4")
        rows[name] = dict(prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                          tree_sha256=hashlib.sha256(json.dumps([json.loads(t) for t in model.get_booster().get_dump(dump_format="json")], sort_keys=True).encode()).hexdigest())
    return dict(input_sha256=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(), results=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args(argv)
    report = runtime_info()
    if args.synthetic:
        report["synthetic"] = sampling_fingerprints()
    else:
        root = Path(__file__).resolve().parent
        sources = sorted((root / "paper_did").glob("*.py")) + [root / "labor_did" / n for n in
                   ["data.py", "industry.py", "config.py", "support.py", "weights.py", "aggregation.py", "io.py"]]
        sources += [root / "ps1_observed.py", root / "observed_acrt.py"]
        report["normalized_source_sha256"] = hashlib.sha256(b"".join(f.read_bytes().replace(b"\r\n", b"\n") for f in sources)).hexdigest()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
