"""Independent fixed-reference PS1/PS2 execution and annual checkpoints."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ATETConfig
from .data import load_data, prepare, prepare_year
from .estimation import EFFECTS, aggregate, pair_scores
from .io import source_hash, versions, write_json
from .kernel_gps import fit_kernel_gps
from .models import fit_nuisance, predict_controls
from .numerics import paper_bandwidth
from .report import write_report

ROOT = Path(__file__).resolve().parents[1]


def reference_coordinate(raw, c):
    return float(np.log(raw*100)) if c.dose_transform == "log_percent" else float(raw)


def reference_frame(frame, raw_reference, c):
    """Apply exclusions before folds, fitting, bandwidth and score evaluation."""
    dc = reference_coordinate(raw_reference,c)
    evaluated = frame if c.sample_exclusion != "analysis" else frame[frame[c.dose] >= dc]
    evaluated = evaluated.reset_index(drop=True)
    cutoff = dc if c.sample_exclusion == "training" else None
    eligible = evaluated if cutoff is None else evaluated[evaluated[c.dose] >= cutoff]
    info = dict(sample_exclusion=c.sample_exclusion,
                full_firms=frame.firm_code.nunique(),full_industries=frame.ind_code.nunique(),
                analysis_firms=evaluated.firm_code.nunique(),analysis_industries=evaluated.ind_code.nunique(),
                training_eligible_firms=eligible.firm_code.nunique(),training_eligible_industries=eligible.ind_code.nunique(),
                target_firms=evaluated.loc[evaluated[c.dose]>dc,"firm_code"].nunique(),
                target_industries=evaluated.loc[evaluated[c.dose]>dc,"ind_code"].nunique())
    info["excluded_analysis_industries"] = info["full_industries"]-info["analysis_industries"]
    info["excluded_training_industries"] = info["full_industries"]-info["training_eligible_industries"]
    return evaluated,replace(c,training_min_dose=cutoff),info


def compute_reference_year(frame, raw_reference, year, c, folder):
    folder.mkdir(parents=True,exist_ok=True)
    frame, fitting_config, sample = reference_frame(frame,raw_reference,c)
    dc = reference_coordinate(raw_reference,c)
    industry_dose = frame.groupby("ind_code")[c.dose].first().to_numpy()
    h = paper_bandwidth(industry_dose,c.bandwidth_factor)
    print(f"{year} | PS{c.psmethod} | reference={100*raw_reference:g}% | {c.sample_exclusion} | analysis G={sample['analysis_industries']}, training G={sample['training_eligible_industries']}",flush=True)
    fitted = fit_nuisance(frame,fitting_config)
    # Mu is needed at a single fixed comparison dose, not all target doses.
    _, predictions = next(predict_controls(frame,np.array([dc]),fitted,fitting_config))
    mu = predictions[0]
    targets = np.sort(frame.loc[frame[c.dose]>dc,c.dose].unique())
    kernel = fit_kernel_gps(frame,np.unique(np.r_[dc,targets]),h,fitting_config) if c.psmethod == 2 else None
    results, endpoint_rows = {}, []
    for dt in targets:
        results[float(dt)] = pair_scores(frame,float(dt),dc,h,mu,fitted,c,kernel=kernel)
        for effect in EFFECTS:
            endpoint_rows.append(dict(year=year,reference_raw=raw_reference,reference_percent=raw_reference*100,
                reference_model=dc,target_model=float(dt),psmethod=c.psmethod,effect=effect,**results[float(dt)][0][effect]))
    endpoints = pd.DataFrame(endpoint_rows)
    tables, scores = [], {}
    for policy in ["strict","supported"]:
        table, vector = aggregate(frame,results,dc,c,year,policy)
        table["reference_raw"],table["reference_percent"],table["reference_model"] = raw_reference,raw_reference*100,dc
        table["sample_exclusion"],table["bandwidth"] = c.sample_exclusion,h
        table["psmethod"] = c.psmethod
        table["analysis_industries"],table["training_eligible_industries"] = sample["analysis_industries"],sample["training_eligible_industries"]
        table["target_population_fraction"] = table.n_target_firms/sample["full_firms"]
        tables.append(table)
        scores[policy] = vector
    table = pd.concat(tables,ignore_index=True)
    table.to_csv(folder/"estimates.csv",index=False,encoding="utf-8-sig")
    endpoints.to_csv(folder/"atet_by_dose.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(folder/"cluster_scores.npz",industries=np.sort(frame.ind_code.unique()),**scores)
    audit = dict(year=year,psmethod=c.psmethod,reference_raw=raw_reference,reference_model=dc,bandwidth=h,
                 kernel_halfwidth=float(np.sqrt(5)*h),sample=sample,nuisance=fitted["audit"],
                 aggregation="empirical firm frequency among D > reference; fixed weights/support in inference",
                 variance="sum reference-pair industry score contributions before squaring; normal intervals; no bootstrap",
                 ess_filter=None)
    if kernel is not None:
        audit["kernel_gps"] = kernel["audit"]
    write_json(folder/"audit.json",audit)
    return table,endpoints,audit


def run(config=None, output_dir=None):
    c = config if isinstance(config,ATETConfig) else ATETConfig(**(config or {}))
    if c.training_min_dose is not None:
        raise ValueError("Set sample_exclusion instead of internal training_min_dose")
    c = c.resolve_models(ROOT)
    c = replace(c,data_path=str((ROOT/c.data_path).resolve()),cache_dir=str((ROOT/c.cache_dir).resolve()))
    raw,digest = load_data(c.data_path,c.cache_dir)
    key = hashlib.sha256(json.dumps([c.to_dict(),digest,source_hash(),versions()],sort_keys=True).encode()).hexdigest()
    out = Path(output_dir).resolve() if output_dir is not None else ROOT/"results"/f"{c.label}_PS{c.psmethod}_{c.sample_exclusion}_{key[:10]}"
    out.mkdir(parents=True,exist_ok=True)
    status_file = out/"status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status["run_key"] != key:
            raise FileExistsError("Output belongs to different settings/code/data; choose a new --out or omit --out")
        if status["status"] == "completed":
            return pd.read_csv(out/"summary.csv"),out
    status = dict(status="running",run_key=key,psmethod=c.psmethod,source_sha256=source_hash(),data_sha256=digest,versions=versions())
    write_json(out/"config.json",c.to_dict())
    write_json(status_file,status)
    try:
        panel,sample,legacy = prepare(raw,c)
        write_json(out/"sample.json",sample)
        frames = {y:prepare_year(panel,y,legacy) for y in c.years}
        tables,endpoints,audits = [],[],[]
        for ri,reference in enumerate(c.reference_doses):
            for year,frame in frames.items():
                folder = out/f"reference_{ri+1}"/str(year)
                marker = folder/"completed.json"
                if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["run_key"] == key:
                    table = pd.read_csv(folder/"estimates.csv")
                    endpoint = pd.read_csv(folder/"atet_by_dose.csv")
                    audit = json.loads((folder/"audit.json").read_text(encoding="utf-8"))
                else:
                    table,endpoint,audit = compute_reference_year(frame,reference,year,c,folder)
                    write_json(marker,dict(run_key=key))
                tables.append(table)
                endpoints.append(endpoint)
                audits.append(audit)
                pd.concat(tables,ignore_index=True).to_csv(out/"estimates.partial.csv",index=False,encoding="utf-8-sig")
        table = pd.concat(tables,ignore_index=True)
        selected = table[table.support_policy.eq(c.support_policy)].copy()
        table.to_csv(out/"estimates.csv",index=False,encoding="utf-8-sig")
        selected.to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
        pd.concat(endpoints,ignore_index=True).to_csv(out/"atet_by_dose.csv",index=False,encoding="utf-8-sig")
        write_json(out/"audit.json",dict(sample=sample,references_years=audits))
        write_report(selected,sample,c,out)
        status.update(status="completed",failed_estimates=int(selected.status.ne("completed").sum()))
        write_json(status_file,status)
        (out/"estimates.partial.csv").unlink(missing_ok=True)
        return selected,out
    except Exception as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(status_file,status)
        raise
