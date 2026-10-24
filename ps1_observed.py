"""Standalone psmethod=1 observed-dose ACRT with annual checkpoints."""

import hashlib
import html
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.data import load_data, prepare_year
from labor_did.io import versions, write_json
from labor_did.support import scopes
from observed_acrt import PRIMARY, aggregate_results, direct_weights, evaluate_pairs
from paper_did.config import PaperConfig
from paper_did.models import fit_nuisance, source_hash
from paper_did.numerics import paper_bandwidth
from paper_did.pipeline import ROOT, plots, prepare


def implementation_hash():
    h = hashlib.sha256(source_hash().encode())
    for name in ["ps1_observed.py", "observed_acrt.py"]:
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def compute_observed_year(frame, year, c, folder):
    frame = frame.reset_index(drop=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    doses = np.sort(frame[c.dose].unique())
    industry_dose = frame.groupby("ind_code")[c.dose].first().to_numpy()
    h = paper_bandwidth(industry_dose, c.bandwidth_factor)
    print(f"{year}: {len(doses)} observed doses; GPS {c.gps_model} / Mu {c.mu_model}", flush=True)
    fitted = fit_nuisance(frame, c)
    local = [(float(d), float(d-c.diff_step)) for d in doses]
    fixed = [np.quantile(industry_dose, p) if c.pair_scale == "quantile" else p for p in c.atet_pairs]
    fixed = [(float(p[1]), float(p[0])) for p in fixed]
    if len(set(fixed)) != len(fixed) or any(dc >= dt for dt, dc in fixed):
        raise ValueError("ATET pairs collapse to identical observed doses")
    results = evaluate_pairs(frame, local + fixed, fitted, c, h)
    targets, available, availability = {}, {}, []
    for scope, sub in scopes(frame).items():
        targets[scope] = set(sub[c.dose].unique())
        available[scope] = set()
        for d in sorted(targets[scope]):
            r = results[(float(d), float(d-c.diff_step))][0][scope]
            valid = r["status"] == "completed" and np.isfinite([r["estimate"], r["se"]]).all()
            if valid:
                available[scope].add(d)
            availability.append(dict(scope=scope, dose=d, firms=sub.loc[sub[c.dose].eq(d), "firm_code"].nunique(),
                                     computable=bool(valid), error=r.get("error", "")))
    tables, contributions = [], {}
    for policy, selected in [("supported", available), ("strict", targets)]:
        t, v = aggregate_results(frame, results, c, year, h, selected)
        t["support_policy"] = policy
        t["psmethod"] = 1
        t["dose_step"], t["bandwidth"], t["kernel_halfwidth"] = c.diff_step, h, np.sqrt(5)*h
        computable = {scope: direct_weights(sub, c.dose, available[scope])[1] for scope, sub in scopes(frame).items()}
        computable["moderation"] = dict(n_used_firms=sum(computable[s]["n_used_firms"] for s in ["sme", "large"]),
                                       coverage=min(computable[s]["coverage"] for s in ["sme", "large"]))
        t["computable_firms"] = t.effect.map({k: d["n_used_firms"] for k, d in computable.items()})
        t["computable_coverage"] = t.effect.map({k: d["coverage"] for k, d in computable.items()})
        t["n_requested_firms"] = t.n_used_firms
        failed = t.status.ne("completed")
        t.loc[failed, ["n_used_firms", "coverage"]] = 0
        tables.append(t)
        contributions[policy] = v
    table = pd.concat(tables, ignore_index=True)
    endpoints = []
    for dt, dc in local:
        rs, _ = results[(dt, dc)]
        for effect in PRIMARY:
            endpoints.append(dict(dose=dt, dcontrol=dc, effect=effect, **{k: rs[effect].get(k, np.nan) for k in
                                  ["estimate", "se", "p_value", "status", "error", "ess_treat", "ess_control", "max_industry_weight", "n_trimmed"]}))
    atet = [dict(year=year, estimand="ATET", pair_id=i, effect=effect, dtreat=dt, dcontrol=dc,
                 **results[(dt, dc)][0][effect]) for i, (dt, dc) in enumerate(fixed) for effect in PRIMARY]
    pd.DataFrame(endpoints).to_csv(folder / "endpoints.csv", index=False)
    pd.DataFrame(availability).to_csv(folder / "availability.csv", index=False)
    pd.DataFrame(atet).to_csv(folder / "atet.csv", index=False)
    table.to_csv(folder / "estimates.csv", index=False)
    np.savez_compressed(folder / "cluster_scores.npz", industries=np.sort(frame.ind_code.unique()), **contributions)
    audit = dict(year=year, psmethod=1, evaluation="observed_doses", interpolation=False, evaluated_doses=len(doses),
                 diff_step=c.diff_step, bandwidth=h, kernel_halfwidth=np.sqrt(5)*h, nuisance=fitted["audit"],
                 ess_filter=None, density_floor=None, n_firms=len(frame), n_industries=len(industry_dose))
    write_json(folder / "audit.json", audit)
    return table, pd.DataFrame(atet), audit


def write_report(table, sample, c, out):
    finite = table[table.estimate.notna()]
    if len(finite):
        plots(finite, out, c)
    images = "".join(f'<img src="{html.escape(p.name)}">' for p in sorted(out.glob("*.png")))
    content = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>PS1 관측점 ACRT</title>'
    content += '<style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #ccc}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    content += f'<h1>{html.escape(c.label)} · PS1 관측점 직접 계산</h1>'
    content += '<p>기업 수로 가중한 전체·SME·Large ACRT와 Large−SME입니다. p값과 구간은 보정 전 점별 값입니다. coverage는 전체/각 규모 집단에서 평균에 포함된 기업 비율입니다. 계산 불가 처치점과 엄격 전체 집계는 estimates.csv 및 years/availability.csv에서 확인할 수 있습니다.</p>'
    cols = ["year", "effect", "estimate", "se", "p_value", "coverage", "n_used_firms", "n_target_firms", "status"]
    content += table[cols].to_html(index=False) + images
    content += '<h2>표본</h2><pre>' + html.escape(json.dumps(sample, ensure_ascii=False, indent=2)) + '</pre>'
    content += '<h2>설정</h2><pre>' + html.escape(json.dumps(c.to_dict(), ensure_ascii=False, indent=2)) + '</pre></html>'
    (out / "report.html").write_text(content, encoding="utf-8")


def run(config=None, output_dir=None, support="supported"):
    if support not in ["supported", "strict"]:
        raise ValueError("support must be supported or strict")
    c = config if isinstance(config, PaperConfig) else PaperConfig(**(config or {}))
    if not c.compute_acrt or c.size_difference != "means":
        raise ValueError("Observed ACRT requires compute_acrt=true and size_difference='means'")
    if c.gps_features != c.mu_industry_features:
        raise ValueError("GPS and Mu industry feature lists must match; edit both lists together")
    c = replace(c, data_path=str((ROOT / c.data_path).resolve()), cache_dir=str((ROOT / c.cache_dir).resolve()))
    out = Path(output_dir or ROOT / "results/ps1_observed").resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, digest = load_data(c.data_path, c.cache_dir)
    code = implementation_hash()
    key = hashlib.sha256(json.dumps([c.to_dict(), digest, code, versions(), support], sort_keys=True).encode()).hexdigest()
    status_file = out / "status.json"
    if status_file.exists():
        old = json.loads(status_file.read_text())
        if old["run_key"] != key:
            raise FileExistsError("Output belongs to different settings/code/data; choose a new --out")
        if old["status"] == "completed":
            return pd.read_csv(out / "summary.csv")
    write_json(out / "config.json", c.to_dict())
    status = dict(status="running", run_key=key, psmethod=1, support_policy=support, source_sha256=code,
                  data_sha256=digest, versions=versions())
    write_json(status_file, status)
    try:
        panel, sample, legacy = prepare(raw, c)
        del raw
        write_json(out / "sample.json", sample)
        tables, atets, audits = [], [], []
        for year in c.years:
            folder = out / "years" / str(year)
            marker = folder / "completed.json"
            if marker.exists() and json.loads(marker.read_text())["run_key"] == key:
                t, a = pd.read_csv(folder / "estimates.csv"), pd.read_csv(folder / "atet.csv")
                audit = json.loads((folder / "audit.json").read_text())
            else:
                t, a, audit = compute_observed_year(prepare_year(panel, year, legacy), year, c, folder)
                write_json(marker, dict(run_key=key))
            tables.append(t)
            atets.append(a)
            audits.append(audit)
            pd.concat(tables, ignore_index=True).to_csv(out / "estimates.partial.csv", index=False)
        table = pd.concat(tables, ignore_index=True)
        chosen = table[table.support_policy.eq(support)].copy()
        table.to_csv(out / "estimates.csv", index=False)
        chosen.to_csv(out / "summary.csv", index=False)
        pd.concat(atets, ignore_index=True).to_csv(out / "atet.csv", index=False)
        write_json(out / "audit.json", dict(sample=sample, years=audits))
        write_report(chosen, sample, c, out)
        status.update(status="completed", failed_estimates=int(chosen.status.ne("completed").sum()))
        write_json(status_file, status)
        (out / "estimates.partial.csv").unlink(missing_ok=True)
        return chosen
    except Exception as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
        write_json(status_file, status)
        raise
