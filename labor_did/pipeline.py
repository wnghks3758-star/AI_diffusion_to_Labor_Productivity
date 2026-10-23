"""Public API with resumable annual checkpoints and complete, unfiltered results."""

import hashlib
import html
import json
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import Config
from .data import load_data, prepare_panel, prepare_year
from .estimation import estimate_year
from .io import versions, write_json
from .plot import plot_event_study

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ["total", "sme", "large", "moderation"]


def holm(p_values, family_size=None):
    values = np.asarray(p_values, float)
    finite = np.isfinite(values)
    clean = values[finite]
    order = np.argsort(clean)
    m = family_size if family_size is not None else len(clean)
    if m < len(clean):
        raise ValueError("The test family cannot be smaller than the observed tests")
    result = np.full(len(values), np.nan)
    adjusted = np.empty_like(clean)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate((m - np.arange(len(clean))) * clean[order])
    )
    result[finite] = adjusted
    return result


def run(config=None, output_dir=None):
    """Run one specification; return its full DataFrame. Paths are project-relative.

    Pass a Config, JSON filename, or a dictionary of overrides. An identical
    completed run is loaded from disk. Interrupted runs resume at the next year.
    """
    if config is None:
        config = Config.load(ROOT / "settings.json")
    elif isinstance(config, (str, Path)):
        config = Config.load(config)
    elif isinstance(config, dict):
        values = Config.load(ROOT / "settings.json").to_dict()
        config = Config.from_dict({**values, **config})
    config = replace(
        config,
        data_path=str((ROOT / config.data_path).resolve()),
        cache_dir=str((ROOT / config.cache_dir).resolve()),
    )
    output = Path(
        output_dir or ROOT / "results" / f"{config.label}_{config.fingerprint}"
    )
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    status = {
        "status": "running",
        "label": config.label,
        "config_id": config.fingerprint,
    }
    tables, endpoints, audits = [], [], []
    try:
        raw, digest = load_data(config.data_path, config.cache_dir)
        source = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest()
        run_key = hashlib.sha256(
            json.dumps(
                [config.fingerprint, digest, source, versions()], sort_keys=True
            ).encode()
        ).hexdigest()
        status.update(run_key=run_key, data_sha256=digest, versions=versions())
        previous = output / "status.json"
        if previous.exists():
            saved = json.loads(previous.read_text())
            if saved.get("run_key") != run_key:
                raise FileExistsError(
                    f"Output belongs to another configuration or implementation: {output}"
                )
            if saved.get("status") == "completed" and all(
                (output / name).exists()
                for name in [
                    "estimates.csv",
                    "summary.csv",
                    "audit.json",
                    "event_study.png",
                    "report.html",
                ]
            ):
                print(f"[{config.label}] saved result: {output}", flush=True)
                return pd.read_csv(output / "estimates.csv")
        config.save(output / "config.json")
        write_json(output / "status.json", status)
        data, sample = prepare_panel(raw, config)
        del raw
        write_json(output / "sample.json", sample)
        for year in config.years:
            year_dir = output / "years" / str(year)
            checkpoint = year_dir / "completed.json"
            if (
                checkpoint.exists()
                and json.loads(checkpoint.read_text()).get("run_key") == run_key
            ):
                table = pd.read_csv(year_dir / "estimates.csv")
                endpoint = pd.read_csv(year_dir / "endpoints.csv")
                audit = json.loads((year_dir / "audit.json").read_text())
                print(f"[{config.label}] {year}: checkpoint", flush=True)
            else:
                print(f"[{config.label}] {year}: estimating", flush=True)
                frame = prepare_year(data, year, config)
                table, endpoint, audit, state = estimate_year(
                    frame,
                    year,
                    config,
                    on_stage=lambda stage, detail: print(f"  {stage}", flush=True),
                )
                year_dir.mkdir(parents=True, exist_ok=True)
                table.to_csv(year_dir / "estimates.csv", index=False)
                endpoint.to_csv(year_dir / "endpoints.csv", index=False)
                write_json(year_dir / "audit.json", audit)
                if config.save_bootstrap_draws:
                    np.savez_compressed(year_dir / "bootstrap.npz", **state)
                write_json(checkpoint, {"run_key": run_key})
            tables.append(table)
            endpoints.append(endpoint)
            audits.append(audit)
            pd.concat(tables, ignore_index=True).to_csv(
                output / "estimates.partial.csv", index=False
            )
        estimates = pd.concat(tables, ignore_index=True)
        primary = estimates.effect.isin(PRIMARY)
        estimates.loc[primary, "p_holm_within_spec"] = holm(
            estimates.loc[primary, "p_value"], int(primary.sum())
        )
        estimates.to_csv(output / "estimates.csv", index=False)
        pd.concat(endpoints, ignore_index=True).to_csv(
            output / "endpoints.csv", index=False
        )
        write_json(output / "audit.json", {"sample": sample, "years": audits})
        summary = estimates.loc[
            primary,
            [
                "year",
                "effect",
                "estimate",
                "se",
                "p_value",
                "p_holm_within_spec",
                "ci_lower",
                "ci_upper",
                "coverage",
                "bandwidth",
            ],
        ]
        summary.to_csv(output / "summary.csv", index=False)
        plot_event_study(
            estimates,
            output / "event_study.png",
            config.base_year,
            title=f"{config.label} — exploratory, conditional intervals",
        )
        report = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>{html.escape(config.label)}</title><style>
body{{max-width:1200px;margin:32px auto;font:15px system-ui;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}
img{{width:100%}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:16px}}
</style><h1>{html.escape(config.label)}</h1>
<p>기업 {sample["analysis_firms"]:,}개 · 산업 {sample["analysis_industries"]:,}개.
효과는 finite-difference ACR이며 조절효과는 Large − SME입니다.
p_value와 그래프 구간은 명목 추론입니다. p_holm_within_spec은 해당 설정 내 다중검정 보정값입니다.
추론은 학습된 모형·지지구간·가중치에 조건부입니다.</p>
<img src="event_study.png" alt="Event study">
{summary.to_html(index=False, float_format=lambda x: f"{x:.6g}")}
<details><summary>전체 실행 설정</summary><pre>{html.escape(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))}</pre></details>
</html>"""
        (output / "report.html").write_text(report, encoding="utf-8")
        status.update(
            status="completed",
            elapsed_seconds=perf_counter() - started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=source,
        )
        write_json(output / "status.json", status)
        (output / "estimates.partial.csv").unlink(missing_ok=True)
        print(
            summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
            flush=True,
        )
        print(f"\nSaved: {output.resolve()}", flush=True)
        return estimates
    except FileExistsError:
        raise
    except Exception as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        write_json(output / "status.json", status)
        raise
