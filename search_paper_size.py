"""200 paper-method specifications prioritizing size definitions, variables and learners."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import search_paper as engine
import search_paper_batch as runner
from labor_did.config import FIRM_CONTROLS, INDUSTRY_CONTROLS
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT
from search_paper_more import effective_fingerprint

DEST = ROOT / "audit/results/paper_search_1101_1300"
PRIOR_NAMES = ["paper_search_200", "paper_search_201_400", "paper_search_401_600", "paper_search_601_800", "paper_search_801_1100"]
PRIOR = [ROOT / "audit/results" / name for name in PRIOR_NAMES]
SEED = 2026102401
WINDOWS = [[2020], [2021], [2022], [2021, 2022], [2020, 2021, 2022]]


def nuisance_configs():
    rng = np.random.default_rng(SEED)
    pairs = [("spline", "xgboost"), ("spline", "ridge"), ("random_forest", "lasso"),
             ("lasso", "random_forest"), ("ridge", "linear"), ("linear", "spline"),
             ("spline", "xgboost"), ("random_forest", "ridge"), ("xgboost", "lasso"), ("lasso", "xgboost")]
    configs = []
    for i, (gps, mu) in enumerate(pairs):
        c = PaperConfig().to_dict()
        for field, pool in [("gps_features", INDUSTRY_CONTROLS), ("mu_industry_features", INDUSTRY_CONTROLS),
                            ("mu_firm_features", FIRM_CONTROLS)]:
            c[field] = [v for v in pool if rng.random() < [.45, .65, .85][i % 3]] or [pool[0]]
        if i in [0, 4, 7, 8]:
            c["mu_firm_features"] += ["large_dummy", "small_dummy"] if i in [0, 7] else ["large_dummy"] if i == 4 else ["small_dummy"]
        if i == 0:
            c.update(gps_features=PaperConfig().gps_features, mu_industry_features=INDUSTRY_CONTROLS.copy(),
                     mu_firm_features=PaperConfig().mu_firm_features)

        def params(model):
            if model == "random_forest":
                return dict(n_estimators=150, max_depth=3, min_samples_leaf=8)
            if model == "xgboost":
                return dict(n_estimators=150, max_depth=2, learning_rate=.05, min_child_weight=7)
            if model in ["spline", "ridge"]:
                return dict(alpha=[.1, 1., 10.][i % 3])
            if model == "lasso":
                return dict(alpha=[.0005, .003, .015][i % 3])
            return {}

        c.update(gps_model=gps, mu_model=mu, gps_params=params(gps), mu_params=params(mu), seed=SEED + i,
                 include_firm_controls=i != 5, bandwidth_factor=.7, trim=.1, n_splits=3,
                 industry_min_groups=4, industry_min_obs=4, size_missing_policy="unclassified")
        if not c["include_firm_controls"]:
            c["mu_firm_features"] = []
        configs.append(c)
    return configs


def make_plan(dest=DEST, prior=PRIOR):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    plans = [json.loads((Path(p) / "plan.json").read_text()) for p in prior]
    summary = json.loads((Path(prior[-1]) / "summary.json").read_text())
    assert summary["cumulative_attempted"] == 1100
    old = [e for p in plans for e in p["entries"]]
    assert len(old) == 1100
    seen = {effective_fingerprint(e["config"]) for e in old}
    profiles = [(v, cutoff, years) for v in ["sale", "assets_annual"] for cutoff in ["median", "mean"] for years in WINDOWS]
    entries = []
    for family, base in enumerate(nuisance_configs()):
        for variant, (variable, cutoff, years) in enumerate(profiles):
            number = 1101 + len(entries)
            time_stat = "median" if len(years) > 1 and family % 2 else "mean"
            c = PaperConfig(**{**base, "size_variable": variable, "size_threshold": cutoff, "size_years": years,
                               "size_time_aggregation": time_stat, "label": f"paper_search_{number}"})
            key = effective_fingerprint(c.to_dict())
            assert key not in seen, f"Duplicate specification: {number}"
            seen.add(key)
            entries.append(dict(number=number, family=family, family_name=f"variables_models_{family + 1}", variant=variant,
                                config=c.to_dict(), fingerprint=c.fingerprint, effective_fingerprint=key))
    assert len(entries) == 200
    plan = {k: plans[-1][k] for k in ["hypothesis_estimand", "alpha", "retention", "criteria", "focused_filter"]}
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), count=200, seed=SEED, entries=entries, size_search=True,
                runner_sha256=file_hash(runner.__file__), planner_sha256=file_hash(__file__), engine_sha256=file_hash(engine.__file__),
                source_sha256=source_hash(), previous_source_sha256=plans[-1]["source_sha256"],
                runner_file="search_paper_batch.py", planner_file="search_paper_size.py",
                data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]), previous_summary=summary,
                previous_count=1100, cumulative_count=1300, correction_family_size=4000, cumulative_correction_family_size=26000,
                prior_plan_sha256=[file_hash(Path(p) / "plan.json") for p in prior],
                previous_summary_sha256=file_hash(Path(prior[-1]) / "summary.json"), screening_order=list(engine.SCREEN_ORDER),
                stop_rule="Immediately stop failed 2023/2020/2021 gates and delete rejected results; attempt all 200.",
                report_stem="Paper_Search_1101_1300", report_title_ko="논문 방식: 규모 구분·변수·학습기 중심 추가 200개",
                design_description_ko="기존 1,100개와 중복 없는 1101~1300번 200개를 실행 전에 확정했습니다. 10개 변수·학습기 조합 각각에 규모 정의 20개를 적용합니다. 기존 코드도 이미 sale을 사용했습니다. 기본 규모 정의를 재현하는 검사를 통과한 뒤 변수·기간·요약 기준을 설정으로 확장했으며 ATET 점수·추론 공식은 유지했습니다.",
                variables_description_ko="산업 변수 13개와 기업 통제 3개를 개별 포함 확률 0.45/0.65/0.85로 선택합니다. 규모 더미도 두 개 모두·하나만·모두 제외를 비교합니다. GPS에는 산업 변수만, Mu에는 산업·선택 기업 변수와 D를 사용합니다. seed 2026102401입니다.",
                models_description_ko="GPS/Mu 조합 10개에 Spline·Random Forest·Lasso·Ridge·선형·XGBoost가 포함됩니다. 규모 비교를 우선하여 Fold 3·산업 분류 최소 산업수/관측수 4·bandwidth 배율 0.70·trim 0.10을 고정했습니다. 조합별 학습 seed와 변수 목록은 계획에 기록했습니다.",
                numeric_description_ko="규모 변수는 sale과 연도별 assets_annual, 산업 내 기준은 median/mean, 기간은 2020년·2021년·2022년 단독 및 2021~2022년·2020~2022년입니다. 기업별 기간 요약은 복수 연도에서 mean/median을 비교합니다. 단일 연도에서는 두 요약이 같아 mean으로 고정합니다. 산업 내 기준은 기업별 요약값을 기업당 한 표로 계산하며 기준 이상은 Large입니다. 규모값이 없거나 산업 내 기업이 1개인 기업은 전체 추정에 남기고 규모 비교에서 미분류합니다. 분류는 모든 분석연도에 고정합니다.",
                correction_note_ko="이번 200개×20개=4,000개, 논문 방식 누적 1,300개×20개=26,000개 검정으로 보정합니다. 규모 정의를 바꾸면 SME/Large의 구성과 추정 대상도 바뀝니다. 이전 결과를 보며 수행한 탐색이므로 원래 p값 또는 이 보정만으로 확증적 결론을 내릴 수 없습니다.")
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
