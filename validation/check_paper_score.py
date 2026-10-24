"""Export actual-data frozen nuisances for the official R score verifier.

Compares score algebra, not whole-package ML. Firm-level fixtures stay in ignored
results/; only aggregate comparison.csv needs to be shared.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labor_did.data import load_data, prepare_year
from paper_did.config import PaperConfig
from paper_did.models import fit_nuisance, predict_controls
from paper_did.numerics import atet_score, np_kernel, paper_bandwidth
from paper_did.pipeline import prepare


def export(c, out):
    c = replace(
        c,
        data_path=str((ROOT / c.data_path).resolve()),
        cache_dir=str((ROOT / c.cache_dir).resolve()),
    )
    raw, _ = load_data(c.data_path, c.cache_dir)
    panel, _, legacy = prepare(raw, c)
    year = 2023 if 2023 in c.years else c.years[0]
    frame = prepare_year(panel, year, legacy).reset_index(drop=True)
    D = frame.groupby("ind_code")[c.dose].first().to_numpy()
    h = paper_bandwidth(D, c.bandwidth_factor)
    dc, dt = np.quantile(D, [0.25, 0.75])
    center = float(np.quantile(D, 0.5))
    pairs = [("central_atet", dt, dc), ("local_acrt", center, center - c.diff_step)]
    controls = np.sort([p[2] for p in pairs])
    fit = fit_nuisance(frame, c)
    predictions = {}
    for start, batch in predict_controls(frame, controls, fit, c):
        for k, mu in enumerate(batch):
            predictions[controls[start + k]] = mu
    rows = []
    expected = []
    for name, t, d in pairs:
        for scope, mask in [
            ("total", np.ones(len(frame), bool)),
            ("sme", frame.large_firm.eq(0).to_numpy()),
            ("large", frame.large_firm.eq(1).to_numpy()),
        ]:
            ix = np.flatnonzero(mask)
            mean = fit["mean"][ix]
            var = fit["variance"][ix]
            p_t = np.exp(-0.5 * (t - mean) ** 2 / var)
            p_c = np.exp(-0.5 * (d - mean) ** 2 / var)
            if (p_t <= 0).any() or (p_c <= 0).any():
                raise ValueError("Ordinary R density underflow in reference fixture")
            sub = frame.iloc[ix]
            high = np_kernel(sub[c.dose], t, h)
            low = np_kernel(sub[c.dose], d, h)
            mu = predictions[d][ix]
            got = atet_score(
                sub.Y_diff,
                mu,
                high,
                low,
                np.log(p_t / p_c),
                np.arange(len(sub)),
                c.trim,
            )
            case = name + "_" + scope
            rows.append(
                pd.DataFrame(
                    dict(
                        case=case,
                        high=high,
                        low=low,
                        pd=p_t,
                        pc=p_c,
                        y=sub.Y_diff.to_numpy(),
                        mu=mu,
                        trim=c.trim,
                    )
                )
            )
            expected.append(
                dict(
                    case=case,
                    Python_ATET=got["atet"],
                    Python_trimmed=got["n_trimmed"],
                    Python_se_iid=got["se"],
                )
            )
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out / "fixture.csv", index=False)
    pd.DataFrame(expected).to_csv(out / "python.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results/paper_r_reference")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--config", type=Path, default=ROOT / "paper_settings.json")
    args = p.parse_args()
    if not args.compare:
        export(PaperConfig.load(args.config), args.out)
    else:
        a = pd.read_csv(args.out / "python.csv")
        b = pd.read_csv(args.out / "r.csv")
        m = a.merge(b, on="case", validate="one_to_one")
        if len(m) != len(a) or len(m) != len(b):
            raise ValueError("Incomplete R comparison")
        m["atet_abs_diff"] = abs(m.Python_ATET - m.R_ATET)
        m["se_abs_diff"] = abs(m.Python_se_iid - m.R_se)
        if (
            not (m.Python_trimmed == m.R_trimmed).all()
            or m.atet_abs_diff.max() > 1e-11
            or m.se_abs_diff.max() > 1e-11
        ):
            raise ValueError("R score mismatch")
        m.to_csv(args.out / "comparison.csv", index=False)
        print(m.to_string(index=False))


if __name__ == "__main__":
    main()
