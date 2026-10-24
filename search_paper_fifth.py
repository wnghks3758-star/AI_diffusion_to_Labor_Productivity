"""Freeze 300 new paper-method configurations, 801--1100, before estimation."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import search_paper as engine
import search_paper_batch as runner
from labor_did.config import INDUSTRY_CONTROLS
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT
from search_paper_more import effective_fingerprint, toggle

DEST = ROOT / "audit/results/paper_search_801_1100"
PRIOR = [
    ROOT / "audit/results" / name
    for name in [
        "paper_search_200",
        "paper_search_201_400",
        "paper_search_401_600",
        "paper_search_601_800",
    ]
]
SEED = 2026102307
CENTERS = [402, 432, 613, 616, 751, 752]
VARIANTS = [(0.40, 0.025), (0.55, 0.05), (0.70, 0.10), (0.95, 0.10), (1.25, 0.15)]


def make_plan(dest=DEST, prior=PRIOR):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    plans = [json.loads((Path(p) / "plan.json").read_text()) for p in prior]
    summary = json.loads((Path(prior[-1]) / "summary.json").read_text())
    assert summary["cumulative_attempted"] == 800
    old = {e["number"]: e for p in plans for e in p["entries"]}
    assert len(old) == 800
    seen = {effective_fingerprint(e["config"]) for e in old.values()}
    rng = np.random.default_rng(SEED)
    families, family_keys = [], set()
    firm_pool = PaperConfig().mu_firm_features
    for number in CENTERS:
        center = old[number]["config"]
        ops = [
            (field, v)
            for field in ["gps_features", "mu_industry_features"]
            for v in INDUSTRY_CONTROLS
        ]
        ops += [("single_firm_control", v) for v in firm_pool]
        count = 0
        for ix in rng.permutation(len(ops)):
            field, variable = ops[ix]
            if field == "single_firm_control":
                c = copy.deepcopy(center)
                c.update(include_firm_controls=True, mu_firm_features=[variable])
            else:
                if center[field] == [variable]:
                    continue
                c = toggle(center, field, variable, INDUSTRY_CONTROLS)
            key = effective_fingerprint({**c, "bandwidth_factor": 0.7, "trim": 0.1})
            if key in family_keys or any(
                effective_fingerprint({**c, "bandwidth_factor": b, "trim": t}) in seen
                for b, t in VARIANTS
            ):
                continue
            family_keys.add(key)
            families.append((f"paper_search_{number}:{field}:{variable}", c))
            count += 1
            if count == 5:
                break
        assert count == 5
    models = ["spline", "random_forest", "lasso", "ridge", "xgboost", "linear"]
    for i in range(30):
        c = PaperConfig().to_dict()
        for field, pool in [
            ("gps_features", INDUSTRY_CONTROLS),
            ("mu_industry_features", INDUSTRY_CONTROLS),
            ("mu_firm_features", firm_pool),
        ]:
            c[field] = [
                v for v in pool if rng.random() < [0.35, 0.55, 0.75, 0.90][i % 4]
            ] or [pool[0]]
        gps, mu = models[i % 6], models[(i + i // 6 + 2) % 6]

        def params(model):
            if model == "random_forest":
                return dict(
                    n_estimators=150,
                    max_depth=[2, 3, 4, 5][i % 4],
                    min_samples_leaf=[5, 8, 12][i % 3],
                )
            if model == "xgboost":
                return dict(
                    n_estimators=[150, 250][i % 2],
                    max_depth=[2, 3][i % 2],
                    learning_rate=[0.025, 0.05][i % 2],
                    min_child_weight=[7, 12][i % 2],
                )
            if model == "lasso":
                return dict(alpha=[0.0005, 0.003, 0.015, 0.05][i % 4])
            if model == "ridge":
                return dict(alpha=[0.03, 0.3, 3.0, 30.0][i % 4])
            return {}

        c.update(
            gps_model=gps,
            mu_model=mu,
            gps_params=params(gps),
            mu_params=params(mu),
            n_splits=[3, 4, 5][i % 3],
            seed=[23, 107, 2029, 419][i % 4],
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
            number = 801 + len(entries)
            c = PaperConfig(
                **{
                    **base,
                    "bandwidth_factor": bw,
                    "trim": trim,
                    "label": f"paper_search_{number}",
                }
            )
            key = effective_fingerprint(c.to_dict())
            assert key not in seen, f"Duplicate: {name}"
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
    assert len(entries) == 300
    plan = {
        k: plans[-1][k]
        for k in [
            "hypothesis_estimand",
            "alpha",
            "retention",
            "criteria",
            "focused_filter",
        ]
    }
    plan.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        count=300,
        seed=SEED,
        entries=entries,
        runner_sha256=file_hash(runner.__file__),
        planner_sha256=file_hash(__file__),
        engine_sha256=file_hash(engine.__file__),
        source_sha256=source_hash(),
        runner_file="search_paper_batch.py",
        planner_file="search_paper_fifth.py",
        data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]),
        previous_summary=summary,
        previous_count=800,
        cumulative_count=1100,
        prior_plan_sha256=[file_hash(Path(p) / "plan.json") for p in prior],
        previous_summary_sha256=file_hash(Path(prior[-1]) / "summary.json"),
        correction_family_size=6000,
        cumulative_correction_family_size=22000,
        screening_order=list(engine.SCREEN_ORDER),
        stop_rule="Immediately stop failed 2023/2020/2021 gates and delete rejected results; attempt all 300.",
        report_stem="Paper_Search_801_1100",
        report_title_ko="논문 방식 추가 300개 탐색: 801~1100번",
        design_description_ko="이전 논문 방식 800개와 중복 없는 300개를 실행 전에 확정했습니다. 같은 논문 방식에서 SME 2025가 양수였던 402·432·613·616·751·752번 주변 30개 조합과 새로운 조합 30개를 각 5개 수치 설정으로 비교했습니다. 이전 양수 추정치들은 유의하지 않았으며, 결과를 보고 정한 탐색입니다. 이전 추정 방식의 결과는 이번 보고서에 섞지 않습니다.",
        variables_description_ko="GPS/Mu 산업 변수의 개별 추가·제외와 기업 변수 개별 선택을 포함합니다. 새 조합의 개별 포함 확률은 0.35/0.55/0.75/0.90이고 seed는 2026102307입니다. 기업 통제 전체 제외도 포함합니다.",
        models_description_ko="GPS·Mu에 Spline/RF/Lasso/Ridge/XGBoost/선형, Fold 3/4/5, seed·산업 더미·산업 분류 최소 관측수·모형 파라미터를 변경했습니다. psmethod=1·2022년 기준 통제·산업 Fold·군집 추론과 ESS 진단만 사용하는 원칙은 유지했습니다.",
        numeric_description_ko="(bandwidth 배율, trim): (0.40,0.025), (0.55,0.05), (0.70,0.10), (0.95,0.10), (1.25,0.15). 산업 수와 처치의 표본 표준편차를 사용하는 저자 bandwidth 공식·추정식·유한차분 간격 0.0005는 유지했습니다.",
        correction_note_ko="300개×20개=6,000개 검정의 배치 보정과 논문 방식 1,100개×20개=22,000개 검정의 누적 보정을 분리합니다. 중단된 설정도 포함합니다. 이전 방식의 탐색은 제외하며, 적응적 탐색이므로 확증적 추론으로 해석하지 않습니다.",
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
        print(f"Frozen {len(make_plan(args.out)['entries'])} configurations")
    else:
        plan = json.loads((args.out / "plan.json").read_text())
        assert plan["planner_sha256"] == file_hash(__file__)
        runner.search(args.out, args.workers)
