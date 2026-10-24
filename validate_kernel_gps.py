"""Bounded, outcome-blind psmethod=2 feasibility study at industry level.

python validate_kernel_gps.py --simulations 100
All configurations and failures are saved; no significance search is performed.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from labor_did import Config
from labor_did.data import cluster_bandwidth, load_data, prepare_panel, prepare_year
from labor_did.io import versions, write_json
from labor_did.kernel_gps import atet_from_kernel_scores, fit_kernel_gps
from labor_did.weights import epanechnikov_kernel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "audit/results/psmethod2_151"
QUANTILES = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
PAIRS = [(i, i + 1) for i in range(6)] + [(2, 4), (1, 5), (0, 6)]


def support_row(frame, dt, dc, H, pt, pc):
    # A zero residual isolates support diagnostics; no effect estimates are selected.
    dummy = frame[["ind_code", "office_ratio"]].copy()
    dummy["Y_diff"] = 0.0
    return atet_from_kernel_scores(dummy, dt, dc, H, pt, pc, np.zeros(len(frame)))


def actual_study():
    c = Config.load(ROOT / "settings.json")
    c = replace(
        c,
        data_path=str(ROOT / c.data_path),
        cache_dir=str(ROOT / ".cache"),
        n_repeats=5,
        kernel_convention="np",
    )
    raw, _ = load_data(c.data_path, c.cache_dir)
    panel, sample = prepare_panel(raw, c)
    frame = prepare_year(panel, 2023, c)
    industries = frame.groupby("ind_code", as_index=False).first()
    D = industries[c.dose].to_numpy()
    probe = np.quantile(D, QUANTILES)
    H0 = cluster_bandwidth(frame, c)
    dense = np.linspace(D.min() - 2 * H0, D.max() + 2 * H0, 301)
    centers = np.unique(np.r_[dense, probe])
    rows, support, fold_support = [], [], []
    plans = []
    for factor in [0.75, 1.0, 1.5, 2.0]:
        for name, params, unconditional in [
            ("unconditional", {}, True),
            ("rf_depth2_leaf10", dict(max_depth=2, min_samples_leaf=10), False),
            ("rf_depth3_leaf8", dict(max_depth=3, min_samples_leaf=8), False),
        ]:
            plans.append((factor, name, params, unconditional, "shared", centers))
    for factor in [0.75, 1.0, 1.5, 2.0]:
        plans.append(
            (
                factor,
                "rf_depth3_leaf8_separate",
                dict(max_depth=3, min_samples_leaf=8),
                False,
                "separate",
                probe,
            )
        )
    baseline = {}
    all_audits = []
    for factor, name, params, unconditional, mode, points in plans:
        start = time.monotonic()
        H = H0 * factor
        fit = fit_kernel_gps(
            frame,
            points,
            H,
            replace(c, gps_params=params),
            mode=mode,
            unconditional=unconditional,
        )
        pidx = np.searchsorted(points, probe)
        target = epanechnikov_kernel(D[:, None], probe[None, :], H)
        pred = fit.predictions[:, :, pidx]
        mse = float(np.mean((pred - target[None, :, :]) ** 2))
        if unconditional:
            baseline[factor] = mse
        r = dict(
            model=name,
            bandwidth_factor=factor,
            h_rule=H / np.sqrt(5),
            physical_halfwidth=H,
            n_industries=len(D),
            n_firms=len(frame),
            kernel_response_mse=mse,
            improvement_vs_unconditional=1 - mse / baseline[factor],
            seconds=time.monotonic() - start,
            zero_prediction_rate=float(np.mean(pred == 0)),
        )
        for j, q in enumerate(QUANTILES):
            r[f"mse_q{q:g}"] = float(np.mean((pred[:, :, j] - target[None, :, j]) ** 2))
        for f in fit.audit["folds"]:
            train = np.asarray(f["train_positions"])
            test = np.asarray(f["test_positions"])
            for j, q in enumerate(QUANTILES):
                fold_support.append(
                    dict(
                        model=name,
                        bandwidth_factor=factor,
                        repetition=f["repetition"],
                        fold=f["fold"],
                        quantile=q,
                        train_industries=len(train),
                        test_industries=len(test),
                        nonzero_training_targets=int(np.sum(target[train, j] > 0)),
                    )
                )
        if mode == "shared":
            # NLL is descriptive; no floor is used in ATET. Exact zeros are separately counted.
            own = np.stack(
                [
                    np.array([np.interp(d, points, curve) for d, curve in zip(D, rep)])
                    for rep in fit.predictions
                ]
            )
            central = (D >= probe[1]) & (D <= probe[-2])
            r.update(
                observed_density_zero_rate=float(np.mean(own <= 0)),
                central80_zero_rate=float(np.mean(own[:, central] <= 0)),
                nll_clipped_1e12=float(-np.log(np.maximum(own, 1e-12)).mean()),
                central80_nll_clipped_1e12=float(
                    -np.log(np.maximum(own[:, central], 1e-12)).mean()
                ),
                max_curve_integral_error=float(
                    np.max(np.abs(np.trapezoid(fit.predictions, points, axis=2) - 1))
                ),
            )
        for rep in range(c.n_repeats):
            scores = fit.firm_scores(rep)
            for low, high in PAIRS:
                s = dict(
                    model=name,
                    bandwidth_factor=factor,
                    repetition=rep,
                    qcontrol=QUANTILES[low],
                    qtreat=QUANTILES[high],
                    dcontrol=probe[low],
                    dtreat=probe[high],
                )
                try:
                    diag = support_row(
                        frame,
                        probe[high],
                        probe[low],
                        H,
                        scores[:, pidx[high]],
                        scores[:, pidx[low]],
                    )
                    s.update(
                        {
                            k: v
                            for k, v in diag.items()
                            if k
                            in [
                                "ess_treat",
                                "ess_control",
                                "supported",
                                "max_industry_weight",
                                "max_firm_weight",
                                "trimmed_firms",
                            ]
                        },
                        error="",
                    )
                except ValueError as e:
                    s.update(supported=False, error=str(e))
                support.append(s)
        rows.append(r)
        all_audits.append(dict(model=name, factor=factor, **fit.audit))
        pd.DataFrame(rows).to_csv(OUT / "actual_cv.csv", index=False)
        pd.DataFrame(support).to_csv(OUT / "actual_support.csv", index=False)
        pd.DataFrame(fold_support).to_csv(OUT / "training_support.csv", index=False)
        write_json(
            OUT / "actual_audit.json",
            dict(
                sample=sample,
                configuration=c.to_dict(),
                probe_quantiles=QUANTILES,
                probe_values=probe,
                bandwidth_halfwidth_base=H0,
                versions=versions(),
                fits=all_audits,
            ),
        )
        print(
            f"actual {name} factor={factor:g}: relative MSE gain={r['improvement_vs_unconditional']:.3f}, {r['seconds']:.1f}s",
            flush=True,
        )
    return rows


def oracle_kernel_scores(points, H, location):
    # E[K_H(D-d)|X] under the known two-normal-mixture DGP, quadrature over D.
    u, w = np.polynomial.legendre.leggauss(64)
    z = points[None, :, None] + H * u[None, None, :]
    f = np.zeros((len(location), len(points), len(u)))
    for prob, offset, scale in [(0.7, -0.35, 0.45), (0.3, 0.75, 0.8)]:
        t = (z - location[:, None, None] - offset) / scale
        f += prob * np.exp(-t * t / 2) / (np.sqrt(2 * np.pi) * scale)
    return np.sum(f * (0.75 * (1 - u * u) * w)[None, None, :], axis=2)


def simulation_study(n_sims, n_clusters=151):
    rows = []
    points = np.array([-0.5, 0.5])
    for iteration in range(n_sims):
        rng = np.random.default_rng(20261023 + iteration)
        X = rng.normal(size=(n_clusters, 2))
        location = 0.6 * X[:, 0] + 0.3 * X[:, 1]
        mix = rng.random(n_clusters) < 0.7
        D = location + np.where(
            mix, rng.normal(-0.35, 0.45, n_clusters), rng.normal(0.75, 0.8, n_clusters)
        )
        # Common shocks survive outcome differencing, so inference must cluster.
        trend = 0.5 * X[:, 0] - 0.25 * X[:, 1] + rng.normal(0, 0.35, n_clusters)
        Y = (
            D[:, None]
            + 0.4 * D[:, None] ** 2
            + trend[:, None]
            + rng.normal(0, 0.3, (n_clusters, 8))
        )
        frame = pd.DataFrame(
            dict(
                ind_code=np.repeat(np.arange(n_clusters), 8),
                office_ratio=np.repeat(D, 8),
                Y_diff=Y.ravel(),
            )
        )
        folds = list(KFold(3, shuffle=True, random_state=42).split(X))
        mu = np.full(n_clusters, np.nan)
        for train, test in folds:
            design = np.column_stack([np.ones(n_clusters), D, D**2, X])
            beta = np.linalg.lstsq(design[train], Y.mean(axis=1)[train], rcond=None)[0]
            counter = design[test].copy()
            counter[:, 1] = points[0]
            counter[:, 2] = points[0] ** 2
            mu[test] = counter @ beta
        for factor in [0.35, 0.5]:
            h = 2.34 * np.std(D) * n_clusters ** (-0.25) * factor
            H = np.sqrt(5) * h
            target = epanechnikov_kernel(D[:, None], points[None, :], H)
            gps = {
                name: np.full((n_clusters, 2), np.nan)
                for name in ["rf_separate", "unconditional"]
            }
            for train, test in folds:
                gps["unconditional"][test] = target[train].mean(axis=0)
                for j in range(2):
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=3,
                        min_samples_leaf=8,
                        max_features=1.0,
                        n_jobs=1,
                        random_state=42,
                    ).fit(X[train], target[train, j])
                    gps["rf_separate"][test, j] = model.predict(X[test])
            gps["oracle_smoothed"] = oracle_kernel_scores(points, H, location)
            for name, ps in gps.items():
                r = dict(
                    iteration=iteration,
                    n_industries=n_clusters,
                    bandwidth_factor=factor,
                    model=name,
                    true_atet=1.0,
                )
                try:
                    est = atet_from_kernel_scores(
                        frame,
                        0.5,
                        -0.5,
                        H,
                        np.repeat(ps[:, 1], 8),
                        np.repeat(ps[:, 0], 8),
                        np.repeat(mu, 8),
                    )
                    r.update(est, error="")
                    r["covered95"] = (
                        abs(est["atet"] - 1) <= 1.96 * est["se_author_cluster"]
                    )
                except ValueError as e:
                    r.update(error=str(e), covered95=False, supported=False)
                rows.append(r)
        if (iteration + 1) % 10 == 0 or iteration + 1 == n_sims:
            pd.DataFrame(rows).to_csv(OUT / "simulation_draws.csv", index=False)
            print(f"simulation {iteration + 1}/{n_sims}", flush=True)
    df = pd.DataFrame(rows)
    summary = []
    for (name, factor), g in df.groupby(["model", "bandwidth_factor"]):
        valid = g.loc[g.error.eq("")]
        err = valid.atet - 1
        coverage = float(valid.covered95.mean()) if len(valid) else np.nan
        summary.append(
            dict(
                model=name,
                bandwidth_factor=factor,
                attempts=len(g),
                estimable=len(valid),
                bias=err.mean(),
                rmse=np.sqrt(np.mean(err**2)),
                coverage95_among_estimable=coverage,
                coverage_mcse=np.sqrt(coverage * (1 - coverage) / len(valid)),
                median_se=valid.se_author_cluster.median(),
                median_ess_treat=valid.ess_treat.median(),
                median_ess_control=valid.ess_control.median(),
                support_rate=float(g.supported.mean()),
            )
        )
    pd.DataFrame(summary).to_csv(OUT / "simulation_summary.csv", index=False)
    write_json(
        OUT / "simulation_design.json",
        dict(
            n_simulations=n_sims,
            n_industries=n_clusters,
            firms_per_industry=8,
            seeds="20261023 + iteration",
            folds=3,
            true_atet=1.0,
            dose_points=points,
            bandwidth_factors=[0.35, 0.5],
            rf=dict(trees=100, depth=3, min_leaf=8),
            versions=versions(),
            limitations=[
                "Favorable correctly specified outcome model; 2 covariates, equal cluster sizes.",
                "This is a finite-sample stress test, not a proof of causal identification or RF convergence rates.",
            ],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--skip-actual", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if not args.skip_actual:
            actual_study()
        simulation_study(args.simulations)


if __name__ == "__main__":
    main()
