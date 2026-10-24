"""Frozen 1538 method comparison, retaining every method/year regardless of significance."""

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.data import load_data, prepare_year
from labor_did.io import file_hash, write_json
from labor_did.support import averaging_weights, scopes
from observed_acrt import (
    aggregate_results,
    availability,
    evaluate_pairs,
    fit_kernel_oof,
)
from paper_did.config import PaperConfig
from paper_did.models import fit_nuisance, source_hash
from paper_did.numerics import paper_bandwidth
from paper_did.pipeline import ROOT, prepare
from paper_did.support import averaging_config
from search_paper import primary

PARENT = ROOT / "audit/results/paper_search_1501_1700/retained/paper_search_1538"
DEST = ROOT / "results/paper_1538_observed"
RF_PARAMS = dict(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features=.75, random_state=2026102409)


def make_plan(dest=DEST, parent=PARENT):
    dest, parent = Path(dest), Path(parent)
    if (dest / "plan.json").exists():
        raise FileExistsError("Comparison plan is already frozen")
    c = PaperConfig(**json.loads((parent / "config.json").read_text()))
    assert c.diff_step == .0005 and c.fold_settings("gps") == (4, 107) and c.fold_settings("mu") == (3, 73)
    files = {name: file_hash(Path(__file__).with_name(name)) for name in ["compare_1538_observed.py", "observed_acrt.py"]}
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=c.to_dict(), kernel_rf_params=RF_PARAMS,
                parent_path=str(parent.relative_to(ROOT)), parent_config_sha256=file_hash(parent / "config.json"),
                parent_estimates_sha256=file_hash(parent / "estimates.csv"), core_source_sha256=source_hash(), files=files,
                data_sha256=file_hash(ROOT / c.data_path), years=c.years, repeats=1,
                methods=["ps1_grid", "ps1_observed", "ps2_rf_observed"],
                targets=["native_supported", "full_observed_strict", "common_ps1", "common_all"],
                inference="Sum industry influence contributions before SE; raw pointwise p-values; no selection by significance.",
                support="Observed-dose failures are explicit; no observed-point interpolation. Report strict full-sample failure and supported/common target coverage separately.",
                identification_note="Industry GPS covariates vs industry+firm Mu controls and clustered application are inherited; psmethod=2 changes GPS learner to RF too.",
                cache="Kernel GPS excludes outcomes/year from key and is reused only when actual industry covariates/dose match; Mu held fixed between methods within year.",
                zero_policy="Reject negative or nonpositive active GPS; no density floor, clipping, or ESS threshold.")
    write_json(dest / "plan.json", plan)
    return plan


def save_endpoints(results, c, path):
    rows = []
    for (dt, dc), (rs, _) in results.items():
        for scope in ["total", "sme", "large", "moderation"]:
            r = rs[scope]
            rows.append(dict(dose=dt, dcontrol=dc, scope=scope, **{k: r.get(k, np.nan) for k in
                             ["estimate", "se", "p_value", "status", "error", "ess_treat", "ess_control", "max_industry_weight", "n_trimmed"]}))
    pd.DataFrame(rows).to_csv(path, index=False)


def run(dest=DEST, workers=2):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    assert plan["core_source_sha256"] == source_hash()
    for name, digest in plan["files"].items():
        assert digest == file_hash(Path(__file__).with_name(name))
    parent = ROOT / plan["parent_path"]
    assert file_hash(parent / "config.json") == plan["parent_config_sha256"]
    assert file_hash(parent / "estimates.csv") == plan["parent_estimates_sha256"]
    work = dest / ".scratch"
    c = replace(PaperConfig(**plan["config"]), cache_dir=str(work / "cache"), data_path=str(ROOT / plan["config"]["data_path"]))
    assert file_hash(c.data_path) == plan["data_sha256"]
    raw, _ = load_data(c.data_path, c.cache_dir)
    panel, sample, legacy = prepare(raw, c)
    del raw
    write_json(dest / "sample.json", sample)
    path = dest / "progress.json"
    progress = json.loads(path.read_text()) if path.exists() else dict(status="running", completed_years=[])
    if progress["status"] == "completed":
        return progress
    write_json(path, progress)
    rows, availability_rows = [], []
    # Preserve annual checkpoints; cases are not screened by the earlier search gates.
    for year in plan["years"]:
        folder = dest / "years" / str(year)
        if year in progress["completed_years"]:
            rows.append(pd.read_csv(folder / "estimates.csv"))
            availability_rows.append(pd.read_csv(folder / "target_availability.csv"))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        frame = prepare_year(panel, year, legacy).reset_index(drop=True)
        observed = np.sort(frame[c.dose].unique())
        h = paper_bandwidth(frame.groupby("ind_code")[c.dose].first(), c.bandwidth_factor)
        print(f"{year}: {len(observed)} observed dose points; fitting shared 1538 nuisance", flush=True)
        fitted = fit_nuisance(frame, c)
        pairs = [(float(d), float(d-c.diff_step)) for d in observed]
        ind_dose = frame.groupby("ind_code")[c.dose].first().to_numpy()
        fixed_pairs = [np.quantile(ind_dose, p) if c.pair_scale == "quantile" else np.asarray(p) for p in c.atet_pairs]
        fixed = [(float(p[1]), float(p[0])) for p in fixed_pairs]
        centers = np.unique([d for pair in pairs + fixed for d in pair])
        rf = replace(c, gps_model="random_forest", gps_params=plan["kernel_rf_params"])
        print(f"{year}: kernel GPS {len(centers)} targets; means/variance from ps1 are unused in ps2", flush=True)
        kernel = fit_kernel_oof(frame, centers, h, rf, workers)
        write_json(folder / "nuisance_audit.json", fitted["audit"])
        write_json(folder / "kernel_audit.json", kernel["audit"])
        if not (dest / "kernel_gps_oof.npz").exists():
            np.savez_compressed(dest / "kernel_gps_oof.npz", centers=kernel["centers"], industries=kernel["industries"], predictions=kernel["predictions"])
        old_support = pd.read_csv(parent / "years" / str(year) / "computability.csv")
        low, high = float(ind_dose.min()), float(ind_dose.max())
        grid = low + c.diff_step * np.arange(int(np.floor((high-low)/c.diff_step)) + 1)
        if high - grid[-1] > 1e-12:
            grid = np.r_[grid, high]
        np.testing.assert_allclose(grid, old_support.dose.to_numpy(), rtol=0, atol=1e-12)
        supported = {scope: grid[old_support[scope].to_numpy(bool)] for scope in scopes(frame)}
        ps1 = evaluate_pairs(frame, pairs + fixed, fitted, c, h)
        ps2 = evaluate_pairs(frame, pairs + fixed, fitted, c, h, kernel)
        masks1, availability1 = availability(frame, ps1, c, grid, supported)
        masks2, availability2 = availability(frame, ps2, c, grid, supported)
        availability_table = pd.concat([availability1.assign(method="ps1_observed"), availability2.assign(method="ps2_rf_observed")], ignore_index=True).assign(year=year)
        availability_table.to_csv(folder / "target_availability.csv", index=False)
        availability_rows.append(availability_table)
        save_endpoints(ps1, c, folder / "ps1_endpoints.csv")
        save_endpoints(ps2, c, folder / "ps2_endpoints.csv")
        targets = {}
        for scope, sub in scopes(frame).items():
            all_d = set(sub[c.dose].unique())
            common1 = masks1[scope]["grid"] & masks1[scope]["observed"]
            targets[scope] = dict(full_observed_strict=all_d, common_ps1=common1,
                                  common_all=common1 & masks2[scope]["observed"])
        write_json(folder / "target_sets.json", {scope: {name: sorted(d) for name, d in values.items()} for scope, values in targets.items()})
        needed = set()
        for scope, sub in scopes(frame).items():
            for selected in [masks1[scope]["grid"], targets[scope]["common_ps1"], targets[scope]["common_all"]]:
                target = sub[sub[c.dose].isin(selected)]
                w, _ = averaging_weights(target, supported[scope], grid, np.sqrt(5)*h, averaging_config(c))
                for d in supported[scope][w > 0]:
                    needed.add((float(d), float(d-c.diff_step)))
        # Each curve is estimated with all original firms; only its averaging weights are restricted.
        grid_results = evaluate_pairs(frame, sorted(needed), fitted, c, h)
        saved_grid = primary(pd.read_csv(parent / "years" / str(year) / "estimates.csv"))
        native_targets = {scope: masks1[scope]["grid"] for scope in scopes(frame)}
        rebuilt, _ = aggregate_results(frame, grid_results, c, year, h, native_targets, mode="grid", grid=grid, supported=supported)
        order = ["total", "sme", "large", "moderation"]
        np.testing.assert_allclose(rebuilt.set_index("effect").loc[order, ["estimate", "se", "p_value"]],
                                   saved_grid.set_index("effect").loc[order, ["estimate", "se", "p_value"]], rtol=0, atol=1e-9)
        year_rows = [rebuilt.assign(method="ps1_grid", target="native_supported")]
        arrays = {}
        for target_name in ["full_observed_strict", "common_ps1", "common_all", "native_supported"]:
            for method, result, masks in [("ps1_observed", ps1, masks1), ("ps2_rf_observed", ps2, masks2)]:
                chosen = {scope: masks[scope]["observed"] if target_name == "native_supported" else targets[scope][target_name] for scope in scopes(frame)}
                t, v = aggregate_results(frame, result, c, year, h, chosen)
                year_rows.append(t.assign(method=method, target=target_name))
                arrays[f"{method}_{target_name}"] = v
            if target_name in ["common_ps1", "common_all"]:
                chosen = {scope: targets[scope][target_name] for scope in scopes(frame)}
                t, v = aggregate_results(frame, grid_results, c, year, h, chosen, mode="grid", grid=grid, supported=supported)
                year_rows.append(t.assign(method="ps1_grid", target=target_name))
                arrays[f"ps1_grid_{target_name}"] = v
        atet = []
        for method, result in [("ps1_observed", ps1), ("ps2_rf_observed", ps2)]:
            for i, pair in enumerate(fixed):
                rs, _ = result[pair]
                for effect in ["total", "sme", "large", "moderation"]:
                    atet.append(dict(year=year, method=method, pair_id=i, estimand="ATET", effect=effect, dtreat=pair[0], dcontrol=pair[1], **rs[effect]))
        pd.DataFrame(atet).to_csv(folder / "atet.csv", index=False)
        table = pd.concat(year_rows, ignore_index=True)
        table.to_csv(folder / "estimates.csv", index=False)
        np.savez_compressed(folder / "cluster_scores.npz", industries=np.sort(frame.ind_code.unique()), **arrays)
        rows.append(table)
        progress["completed_years"].append(year)
        write_json(path, progress)
        print(f"{year}: completed; Gaussian vs observed and RF kernel comparisons saved", flush=True)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(dest / "estimates.csv", index=False)
    pd.concat(availability_rows, ignore_index=True).to_csv(dest / "target_availability.csv", index=False)
    progress["status"] = "completed"
    write_json(path, progress)
    shutil.rmtree(work, ignore_errors=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.out) if args.command == "plan" else run(args.out, args.workers), ensure_ascii=False))
