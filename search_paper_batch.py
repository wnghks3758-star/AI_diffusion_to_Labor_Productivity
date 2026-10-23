"""Run a frozen, arbitrary-size exploration with the unchanged paper estimator."""

import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import search_paper as engine
from labor_did.io import file_hash, write_json
from paper_did.models import source_hash
from paper_did.pipeline import ROOT


def run_family(entries, dest):
    plan = json.loads((Path(dest) / "plan.json").read_text())
    result = engine.run_family(entries, dest)
    # The original 200-case engine supplies temporary 4,000-test columns.
    # Replace both corrections and their flags before publishing a completed family.
    for label in result["retained"]:
        folder = Path(dest) / "retained" / label
        t = pd.read_csv(folder / "estimates.csv")
        active = t.estimand.eq(engine.ESTIMAND) & t.effect.isin(engine.PRIMARY)
        for column, size in [
            ("p_bonferroni_search", plan["correction_family_size"]),
            ("p_bonferroni_cumulative", plan["cumulative_correction_family_size"]),
        ]:
            t[column] = np.where(active, np.minimum(1.0, t.p_value * size), np.nan)
        t.to_csv(folder / "estimates.csv", index=False)
        engine.primary(t).to_csv(folder / "summary.csv", index=False)
        write_json(
            folder / "adjusted_flags.json", engine.evaluate(t, "p_bonferroni_search")
        )
        write_json(
            folder / "cumulative_flags.json",
            engine.evaluate(t, "p_bonferroni_cumulative"),
        )
    return result


def validate_plan(plan):
    entries = plan["entries"]
    assert len(entries) == plan["count"] > 0
    assert plan["correction_family_size"] == plan["count"] * 20
    assert plan["cumulative_correction_family_size"] == plan["cumulative_count"] * 20
    assert len({e["number"] for e in entries}) == len(entries)
    assert len({e["effective_fingerprint"] for e in entries}) == len(entries)
    assert plan["runner_sha256"] == file_hash(__file__)
    assert plan["engine_sha256"] == file_hash(engine.__file__)
    assert plan["source_sha256"] == source_hash()
    assert plan["data_sha256"] == file_hash(ROOT / entries[0]["config"]["data_path"])


def search(dest, workers=3):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    validate_plan(plan)
    path = dest / "progress.json"
    progress = (
        json.loads(path.read_text())
        if path.exists()
        else dict(completed_families=[], counts={}, retained=[])
    )
    families = {}
    for entry in plan["entries"]:
        if entry["family"] not in progress["completed_families"]:
            families.setdefault(entry["family"], []).append(entry)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_family, entries, dest) for entries in families.values()
        ]
        for future in as_completed(futures):
            r = future.result()
            progress["completed_families"].append(r["family"])
            counts = Counter(progress["counts"])
            counts.update(r["counts"])
            attempted = sum(counts.values())
            assert attempted <= plan["count"]
            progress.update(
                counts=dict(counts),
                retained=sorted(set(progress["retained"] + r["retained"])),
                attempted=attempted,
                updated_utc=datetime.now(timezone.utc).isoformat(),
                status="completed" if attempted == plan["count"] else "running",
            )
            write_json(path, progress)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in progress.items()
                        if k not in ["completed_families", "retained"]
                    }
                ),
                flush=True,
            )
    shutil.rmtree(dest / ".scratch", ignore_errors=True)
    return progress
