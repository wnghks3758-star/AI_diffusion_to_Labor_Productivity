"""Additional, deduplicated specifications 201--400 using the unchanged estimator."""

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import search_paper as engine
from labor_did.config import INDUSTRY_CONTROLS
from labor_did.io import file_hash, versions, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT

DEST = ROOT / "audit/results/paper_search_201_400"
PREVIOUS = ROOT / "audit/results/paper_search_200"
SEED = 2026102304
VARIANTS = [(0.40, 0.05), (0.55, 0.10), (0.70, 0.10), (0.85, 0.05), (1.00, 0.15)]


def effective_fingerprint(values):
    c = PaperConfig(**values).to_dict()
    c.pop("label")
    c.pop("cache_dir")
    if not c["include_firm_controls"]:
        c["mu_firm_features"] = []
    for field in ["gps_features", "mu_industry_features", "mu_firm_features"]:
        c[field] = sorted(c[field])
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def toggle(config, field, variable, pool):
    c = copy.deepcopy(config)
    selected = set(c[field])
    if variable in selected:
        selected.remove(variable)
    else:
        selected.add(variable)
    c[field] = [x for x in pool if x in selected]
    assert c[field]
    return c


def make_plan(dest=DEST, previous=PREVIOUS):
    dest, previous = Path(dest), Path(previous)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    old = json.loads((previous / "plan.json").read_text())
    previous_summary = json.loads((previous / "summary.json").read_text())
    assert previous_summary["attempted"] == 200
    rng = np.random.default_rng(SEED)
    families = []
    for label, choices in [
        (
            "paper_search_191",
            [
                ("gps_features", INDUSTRY_CONTROLS[:5]),
                ("mu_industry_features", INDUSTRY_CONTROLS[:5]),
            ],
        ),
        (
            "paper_search_183",
            [
                ("gps_features", INDUSTRY_CONTROLS[5:9]),
                ("mu_industry_features", INDUSTRY_CONTROLS[5:9]),
                ("mu_firm_features", ["ln_firm_avg_pay", "ln_firm_ast"]),
            ],
        ),
    ]:
        center = json.loads((previous / "retained" / label / "config.json").read_text())
        for field, variables in choices:
            pool = (
                PaperConfig().mu_firm_features
                if field == "mu_firm_features"
                else INDUSTRY_CONTROLS
            )
            for variable in variables:
                families.append(
                    (
                        f"{label}:toggle:{field}:{variable}",
                        toggle(center, field, variable, pool),
                    )
                )
    assert len(families) == 20
    gps_models = ["spline", "random_forest", "lasso", "ridge", "xgboost"]
    mu_models = ["xgboost", "random_forest", "ridge", "lasso", "spline", "linear"]
    for i in range(20):
        values = PaperConfig().to_dict()
        for field, pool in [
            ("gps_features", INDUSTRY_CONTROLS),
            ("mu_industry_features", INDUSTRY_CONTROLS),
            ("mu_firm_features", values["mu_firm_features"]),
        ]:
            selected = [v for v in pool if rng.random() < [0.50, 0.65, 0.80][i % 3]]
            values[field] = selected or [pool[0]]
        gps, mu = gps_models[i % 5], mu_models[i % 6]

        def params(kind):
            if kind == "random_forest":
                return dict(
                    n_estimators=150,
                    max_depth=[2, 3, 4][i % 3],
                    min_samples_leaf=[5, 8, 12][i % 3],
                )
            if kind == "xgboost":
                return dict(
                    n_estimators=[150, 250][i % 2],
                    max_depth=[2, 3][i % 2],
                    learning_rate=[0.03, 0.05][i % 2],
                    min_child_weight=[7, 12][i % 2],
                )
            if kind == "lasso":
                return dict(alpha=[0.001, 0.01, 0.05][i % 3])
            if kind == "ridge":
                return dict(alpha=[0.1, 1.0, 10.0][i % 3])
            return {}

        values.update(
            gps_model=gps,
            mu_model=mu,
            gps_params=params(gps),
            mu_params=params(mu),
            n_splits=[3, 4, 5][i % 3],
            seed=[7, 42, 2026, 91][i % 4],
            spline_n_knots=[3, 4, 5][i % 3],
            spline_degree=[2, 3][i % 2],
            industry_min_groups=[2, 4, 6][i % 3],
            industry_min_obs=[2, 4, 8][i % 3],
            include_industry_dummies=i % 7 != 0,
            industry_scheme="major" if i % 9 == 0 else "refined",
            include_firm_controls=i % 4 != 0,
        )
        if not values["include_firm_controls"]:
            values["mu_firm_features"] = []
        families.append((f"random_individual_subset_{i + 1}", values))
    prior_keys = {effective_fingerprint(e["config"]) for e in old["entries"]}
    prior_prints = {e["fingerprint"] for e in old["entries"]}
    seen = set(prior_keys)
    entries = []
    for family, (name, base) in enumerate(families):
        for variant, (bw, trim) in enumerate(VARIANTS):
            number = 201 + len(entries)
            values = {
                **base,
                "bandwidth_factor": bw,
                "trim": trim,
                "label": f"paper_search_{number:03d}",
            }
            c = PaperConfig(**values)
            key = effective_fingerprint(values)
            assert key not in seen and c.fingerprint not in prior_prints, (
                f"Duplicate: {name}"
            )
            seen.add(key)
            entries.append(
                dict(
                    number=number,
                    family=family,
                    family_name=name,
                    variant=variant,
                    config=c.to_dict(),
                    fingerprint=c.fingerprint,
                    effective_fingerprint=key,
                )
            )
    assert len(entries) == 200
    plan = {
        k: old[k] for k in ["hypothesis_estimand", "alpha", "retention", "criteria"]
    }
    plan.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        count=200,
        seed=SEED,
        source_sha256=source_hash(),
        runner_sha256=file_hash(__file__),
        engine_sha256=file_hash(engine.__file__),
        versions=versions(),
        data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]),
        previous_plan_sha256=file_hash(previous / "plan.json"),
        previous_summary=previous_summary,
        previous_summary_sha256=file_hash(previous / "summary.json"),
        previous_count=200,
        cumulative_count=400,
        correction_family_size=4000,
        cumulative_correction_family_size=8000,
        screening_order=list(engine.SCREEN_ORDER),
        stop_rule="Same gates, screen 2023 then 2020/2021; all 200 unique additional settings are attempted.",
        report_stem="Paper_Search_201_400",
        report_title_ko="논문 방식 기반 추가 200개 탐색: 201~400번",
        design_description_ko="이전 200개와 중복되지 않는 201~400번 설정을 실행 전에 확정했습니다. 이전 191번(Ridge)과 183번(Random Forest) 주변에서 개별 변수를 하나씩 추가·제외하는 20개 조합, 개별 변수·학습기를 새로 조합한 20개에 각각 5개 bandwidth·트리밍을 적용했습니다. 이전 결과를 보고 정한 후속 탐색이므로 누적 탐색 전체가 사전 확정된 설계는 아닙니다.",
        variables_description_ko="이전 191번 주변 10개, 183번 주변 10개는 GPS/Mu의 산업 변수 또는 Mu 기업 변수를 한 개씩 넣거나 뺍니다. 나머지 20개는 산업 변수 13개와 기업 변수 5개를 각각 포함 확률 0.50/0.65/0.80로 추출했습니다(seed 2026102304). 기업 변수 전체 제외도 포함하며 처치 D는 Mu에 유지합니다. 비활성 기업 변수 목록을 정규화해 중복을 검사했습니다.",
        models_description_ko="GPS Spline·Random Forest·Lasso·Ridge·XGBoost와 Mu XGBoost·Random Forest·Ridge·Lasso·Spline·선형을 비교했습니다. Fold 3/4/5, seed 7/42/2026/91, spline knot 3/4/5·차수 2/3, 산업 최소 산업수 2/4/6·최소 관측수 2/4/8, 산업 더미 제외·대분류도 포함했습니다. 나무 수·깊이·정규화 등 정확한 값은 계획과 설정 CSV에 있습니다.",
        numeric_description_ko="각 조합에 적용한 (bandwidth 배율, 트리밍)은 (0.40,0.05), (0.55,0.10), (0.70,0.10), (0.85,0.05), (1.00,0.15)입니다. 산업 처치값 표본 표준편차와 산업 수로 저자 공식을 계산한 뒤 배율을 곱합니다. 저자 기준 배율은 0.70입니다.",
        correction_note_ko="이번 200개×20개 ACRT 검정의 Bonferroni 보정(p_bonferroni_search, 4,000개)과 앞선 200개까지 합한 보정(p_bonferroni_cumulative, 8,000개)을 분리했습니다. 중단한 검정도 분모에서 제외하지 않습니다. 두 배치 이전의 과거 탐색은 누적 8,000개에 포함하지 않습니다. 이전 결과를 보고 후보를 정한 적응적 탐색이며 이 보정만으로 확증적 추론이 정당화되는 것은 아닙니다.",
        entries=entries,
    )
    write_json(dest / "plan.json", plan)
    return plan


def run_family(entries, dest):
    plan_path = Path(dest) / "plan.json"
    cumulative_size = (
        json.loads(plan_path.read_text()).get("cumulative_correction_family_size", 8000)
        if plan_path.exists()
        else 8000
    )
    result = engine.run_family(entries, dest)
    for label in result["retained"]:
        folder = Path(dest) / "retained" / label
        t = pd.read_csv(folder / "estimates.csv")
        active = t.estimand.eq(engine.ESTIMAND) & t.effect.isin(engine.PRIMARY)
        t["p_bonferroni_cumulative"] = np.where(
            active, np.minimum(1.0, t.p_value * cumulative_size), np.nan
        )
        t.to_csv(folder / "estimates.csv", index=False)
        engine.primary(t).to_csv(folder / "summary.csv", index=False)
        write_json(
            folder / "cumulative_flags.json",
            engine.evaluate(t, "p_bonferroni_cumulative"),
        )
    return result


def search(dest=DEST, workers=3):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    assert plan["runner_sha256"] == file_hash(__file__)
    assert plan["engine_sha256"] == file_hash(engine.__file__)
    assert plan["source_sha256"] == source_hash()
    assert plan["data_sha256"] == file_hash(
        ROOT / plan["entries"][0]["config"]["data_path"]
    )
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
            progress.update(
                counts=dict(counts),
                retained=sorted(set(progress["retained"] + r["retained"])),
                attempted=sum(counts.values()),
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            progress["status"] = (
                "completed" if progress["attempted"] == 200 else "running"
            )
            write_json(path, progress)
            print(
                json.dumps(
                    {k: v for k, v in progress.items() if k != "completed_families"}
                ),
                flush=True,
            )
    shutil.rmtree(dest / ".scratch", ignore_errors=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        p = make_plan(args.out)
        print(f"Frozen {len(p['entries'])} additional settings: {args.out}")
    else:
        search(args.out, args.workers)
