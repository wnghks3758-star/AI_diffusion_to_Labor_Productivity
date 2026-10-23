"""Fixed additional exploration 401--600; raw rejected estimates are not retained."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import search_paper as score_engine
import search_paper_more as runner
from labor_did.config import INDUSTRY_CONTROLS
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT

DEST = ROOT / "audit/results/paper_search_401_600"
PREVIOUS = ROOT / "audit/results/paper_search_201_400"
SEED = 2026102305
VARIANTS = [(0.45, 0.05), (0.55, 0.075), (0.65, 0.10), (0.80, 0.075), (1.10, 0.15)]


def make_plan(dest=DEST, previous=PREVIOUS):
    dest, previous = Path(dest), Path(previous)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    old = json.loads((previous / "plan.json").read_text())
    summary = json.loads((previous / "summary.json").read_text())
    assert summary["attempted"] == 200 and summary["cumulative_attempted"] == 400
    first = ROOT / "audit/results/paper_search_200/plan.json"
    prior_plans = [json.loads(first.read_text()), old]
    seen = {
        runner.effective_fingerprint(e["config"])
        for p in prior_plans
        for e in p["entries"]
    }
    families = []
    family_keys = set()
    rng = np.random.default_rng(SEED)
    for label in ["paper_search_212", "paper_search_242"]:
        center = json.loads((previous / "retained" / label / "config.json").read_text())
        ops = [
            (field, v)
            for field in ["gps_features", "mu_industry_features"]
            for v in INDUSTRY_CONTROLS
        ]
        ops += [("single_firm_control", v) for v in PaperConfig().mu_firm_features]
        count = 0
        for ix in rng.permutation(len(ops)):
            field, v = ops[ix]
            if field == "single_firm_control":
                values = copy.deepcopy(center)
                values.update(include_firm_controls=True, mu_firm_features=[v])
            else:
                values = runner.toggle(center, field, v, INDUSTRY_CONTROLS)
            # Numerical coordinates are fixed here only for nuisance-family deduplication.
            key = runner.effective_fingerprint(
                {**values, "bandwidth_factor": 0.7, "trim": 0.1}
            )
            if key in family_keys:
                continue
            family_keys.add(key)
            families.append((f"{label}:{field}:{v}", values))
            count += 1
            if count == 10:
                break
        assert count == 10
    gps_models = ["spline", "random_forest", "lasso", "ridge", "xgboost", "linear"]
    mu_models = ["ridge", "xgboost", "random_forest", "spline", "lasso", "linear"]
    for i in range(20):
        c = PaperConfig().to_dict()
        for field, pool in [
            ("gps_features", INDUSTRY_CONTROLS),
            ("mu_industry_features", INDUSTRY_CONTROLS),
            ("mu_firm_features", c["mu_firm_features"]),
        ]:
            c[field] = [v for v in pool if rng.random() < [0.4, 0.65, 0.9][i % 3]] or [
                pool[0]
            ]

        def params(model):
            if model == "random_forest":
                return dict(
                    n_estimators=150,
                    max_depth=[2, 3, 4, 5][i % 4],
                    min_samples_leaf=[5, 8, 12][i % 3],
                )
            if model == "xgboost":
                return dict(
                    n_estimators=[150, 300][i % 2],
                    max_depth=[2, 3][i % 2],
                    learning_rate=[0.03, 0.05][i % 2],
                    min_child_weight=[7, 10][i % 2],
                )
            if model == "lasso":
                return dict(alpha=[0.0005, 0.005, 0.02][i % 3])
            if model == "ridge":
                return dict(alpha=[0.03, 0.3, 3.0, 30.0][i % 4])
            return {}

        gps = gps_models[i % 6]
        mu = mu_models[(i + i // 6) % 6]
        c.update(
            gps_model=gps,
            mu_model=mu,
            gps_params=params(gps),
            mu_params=params(mu),
            n_splits=[3, 4, 5][i % 3],
            seed=[11, 73, 2027, 112][i % 4],
            spline_n_knots=[3, 4, 5][i % 3],
            spline_degree=[2, 3][i % 2],
            industry_min_groups=[2, 4, 6][i % 3],
            industry_min_obs=[2, 4, 8][i % 3],
            industry_scheme="major" if i % 8 == 0 else "refined",
            include_industry_dummies=i % 7 != 0,
            include_firm_controls=i % 4 != 0,
        )
        if not c["include_firm_controls"]:
            c["mu_firm_features"] = []
        families.append((f"fresh_individual_subset_{i + 1}", c))
    entries = []
    for family, (name, base) in enumerate(families):
        for variant, (bw, trim) in enumerate(VARIANTS):
            number = 401 + len(entries)
            c = PaperConfig(
                **{
                    **base,
                    "bandwidth_factor": bw,
                    "trim": trim,
                    "label": f"paper_search_{number}",
                }
            )
            key = runner.effective_fingerprint(c.to_dict())
            assert key not in seen, f"Duplicate {name}"
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
        entries=entries,
        runner_sha256=file_hash(runner.__file__),
        planner_sha256=file_hash(__file__),
        engine_sha256=file_hash(score_engine.__file__),
        source_sha256=source_hash(),
        runner_file="search_paper_more.py",
        planner_file="search_paper_next.py",
        data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]),
        previous_plan_sha256=file_hash(previous / "plan.json"),
        previous_summary_sha256=file_hash(previous / "summary.json"),
        first_plan_sha256=file_hash(first),
        previous_summary=summary,
        previous_count=400,
        cumulative_count=600,
        correction_family_size=4000,
        cumulative_correction_family_size=12000,
        screening_order=list(score_engine.SCREEN_ORDER),
        stop_rule="Screen 2023 total decline first, then all four pre-period effects in 2020/2021; stop each failed specification; attempt all 200.",
        report_stem="Paper_Search_401_600",
        report_title_ko="논문 방식 기반 추가 200개 탐색: 401~600번",
        design_description_ko="앞선 400개와 중복되지 않는 401~600번 200개를 실행 전에 확정했습니다. 이전에 SME 2025 추정치가 양수였던 212·242번 주변 20개 변수 조합과 새로운 변수·학습기 조합 20개를 각각 5개 수치 설정으로 비교했습니다. 두 기준 설정의 SME 2025 양수는 유의하지 않았으며, 그 점추정치를 보고 후보를 정한 후속 탐색입니다.",
        variables_description_ko="212번과 242번에서 각각 10개 조합을 정했습니다. GPS/Mu 산업 변수를 하나씩 추가·제외하거나 Mu에 기업 변수 하나를 넣는 후보를 고정 seed로 추출했습니다. 나머지 20개 조합은 산업 변수 13개와 기업 변수 5개를 개별 포함 확률 0.40/0.65/0.90으로 추출했습니다(seed 2026102305). 기업 변수 전체 제외도 포함합니다.",
        models_description_ko="GPS와 Mu에서 Spline·Random Forest·Lasso·Ridge·XGBoost·선형을 비교합니다. Fold 3/4/5, seed 11/73/2027/112, spline knot 3/4/5·차수 2/3, 산업 최소 산업수 2/4/6·최소 관측수 2/4/8 및 산업 더미·분류를 바꿨습니다. 세부 모형 파라미터와 개별 변수 목록은 계획과 CSV에 고정했습니다.",
        numeric_description_ko="각 변수·학습기 조합의 (bandwidth 배율, 트리밍)은 (0.45,0.05), (0.55,0.075), (0.65,0.10), (0.80,0.075), (1.10,0.15)입니다. 산업 수와 산업 처치의 표본 표준편차로 저자 bandwidth 공식을 계산한 뒤 배율을 곱했습니다. 저자 기본 배율은 0.70입니다.",
        correction_note_ko="이번 200개×20개 ACRT 검정에 대한 보정(p_bonferroni_search, 4,000개)과 세 배치 600개 설정의 누적 보정(p_bonferroni_cumulative, 12,000개)을 분리합니다. 중단된 설정도 계획된 검정 수에 포함합니다. 이 세 배치 이전의 과거 탐색은 포함하지 않습니다. 이전 결과를 보고 후보를 정한 적응적 탐색이므로 이 보정만으로 확증적 추론을 정당화할 수 없습니다.",
    )
    write_json(dest / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(args.out)
        print(f"Frozen {len(plan['entries'])} configurations: {args.out}")
    else:
        plan = json.loads((args.out / "plan.json").read_text())
        assert plan["planner_sha256"] == file_hash(__file__)
        runner.search(args.out, args.workers)
