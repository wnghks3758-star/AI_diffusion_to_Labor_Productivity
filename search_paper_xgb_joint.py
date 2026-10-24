"""200 XGBoost/XGBoost specifications with shared industry covariates and independent folds."""

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
from search_paper_size_from_passed import PRIOR as EARLIER

DEST = ROOT / "audit/results/paper_search_1501_1700"
PRIOR = [*EARLIER, ROOT / "audit/results/paper_search_1301_1500"]
SEED = 2026102402


def tree_params(rng, role, random_state):
    if role == "gps":
        trees, depth, child = [75, 100, 150, 200], [1, 2, 3], [5, 8, 12, 20]
    else:
        trees, depth, child = [100, 150, 200, 300], [2, 3, 4], [5, 10, 20]
    return dict(n_estimators=int(rng.choice(trees)), max_depth=int(rng.choice(depth)),
                min_child_weight=int(rng.choice(child)), learning_rate=float(rng.choice([.025, .05, .075])),
                subsample=float(rng.choice([.7, .85, 1.])), colsample_bytree=float(rng.choice([.7, .85, 1.])),
                reg_lambda=float(rng.choice([1., 3., 10.])), reg_alpha=float(rng.choice([0., .05, .1])),
                random_state=int(random_state))


def make_plan(dest=DEST, prior=PRIOR):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    prior = list(map(Path, prior))
    plans = [json.loads((p / "plan.json").read_text()) for p in prior]
    summary = json.loads((prior[-1] / "summary.json").read_text())
    assert summary["cumulative_attempted"] == 1500
    old = [e for p in plans for e in p["entries"]]
    assert len(old) == 1500
    seen = {effective_fingerprint(e["config"]) for e in old}
    rng = np.random.default_rng(SEED)
    masks, entries = set(), []
    firm_pool = ["large_dummy", "small_dummy"] + FIRM_CONTROLS
    for family in range(40):
        if family == 0:
            industry = INDUSTRY_CONTROLS.copy()
        elif family == 1:
            industry = []
        else:
            while True:
                industry = [v for v in INDUSTRY_CONTROLS if rng.random() < [.3, .45, .6, .75, .9][family % 5]]
                if tuple(industry) not in masks:
                    break
        assert tuple(industry) not in masks
        masks.add(tuple(industry))
        gps_seed = [1, 7, 23, 107][family % 4]
        gps_splits = [3, 4, 5][family % 3]
        gps_params = tree_params(rng, "gps", SEED + family)
        for variant in range(5):
            probability = [0., .25, .5, .75, 1.][variant]
            firm = [v for v in firm_pool if rng.random() < probability]
            mu_splits = [2, 3, 4, 5][(family + variant) % 4]
            mu_seed = gps_seed if variant == 0 else [19, 73, 127, 2029][(family + variant) % 4]
            number = 1501 + len(entries)
            c = PaperConfig(gps_model="xgboost", mu_model="xgboost", bandwidth_factor=.5, trim=.01,
                            gps_features=industry.copy(), mu_industry_features=industry.copy(),
                            mu_firm_features=firm, include_firm_controls=bool(firm),
                            gps_params=gps_params.copy(), mu_params=tree_params(rng, "mu", SEED + 1000 + family * 5 + variant),
                            gps_n_splits=gps_splits, mu_n_splits=mu_splits,
                            gps_fold_seed=gps_seed, mu_fold_seed=mu_seed, seed=1,
                            size_variable="sale", size_years=None, size_time_aggregation="mean",
                            size_threshold="median", size_missing_policy="legacy", label=f"paper_search_{number}")
            key = effective_fingerprint(c.to_dict())
            assert key not in seen
            seen.add(key)
            assert c.gps_features == c.mu_industry_features
            entries.append(dict(number=number, family=family, family_name=f"joint_industry_gps_{family + 1}", variant=variant,
                                config=c.to_dict(), fingerprint=c.fingerprint, effective_fingerprint=key))
    assert len(entries) == 200
    plan = {k: plans[-1][k] for k in ["hypothesis_estimand", "alpha", "retention", "criteria", "focused_filter"]}
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), count=200, seed=SEED, entries=entries,
                joint_industry_xgb=True, compact_case_plots=True,
                runner_sha256=file_hash(runner.__file__), planner_sha256=file_hash(__file__), engine_sha256=file_hash(engine.__file__),
                source_sha256=source_hash(), previous_source_sha256=plans[-1]["source_sha256"],
                runner_file="search_paper_batch.py", planner_file="search_paper_xgb_joint.py",
                data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]), previous_summary=summary,
                previous_count=1500, cumulative_count=1700, correction_family_size=4000, cumulative_correction_family_size=34000,
                prior_plan_sha256=[file_hash(p / "plan.json") for p in prior], previous_summary_sha256=file_hash(prior[-1] / "summary.json"),
                screening_order=list(engine.SCREEN_ORDER),
                stop_rule="Stop failed 2023/2020/2021 gates immediately and delete rejected estimates, plots and fitted models; attempt all 200.",
                report_stem="Paper_Search_1501_1700", report_title_ko="GPS·Mu XGBoost: 산업 변수 공동 선택·독립 Fold 추가 200개",
                design_description_ko="사용자 지정 조건을 고정한 1501~1700번 200개를 실행 전에 확정했습니다. GPS와 Mu는 모두 XGBoost, bandwidth 배율 0.50, trim 0.01입니다. 규모는 원래 sale·2020~2022년 기업 평균·산업 내 median으로 복원했습니다. 이전 자산 기준을 사용하지 않습니다.",
                variables_description_ko="산업 통제변수 13개는 GPS와 Mu 산업 변수 목록에서 동시에 포함·제외합니다. 전부 포함·전부 제외와 38개 개별 부분집합을 비교합니다. 산업 더미도 두 모델에 동일하게 포함합니다. 기업 변수 5개는 Mu에서 개별 선택하며 전체 제외도 포함합니다. D는 Mu에 별도 추가됩니다.",
                models_description_ko="40개 산업 변수·GPS 설정마다 5개 Mu 설정을 적용했습니다. GPS Fold 3/4/5와 Mu Fold 2/3/4/5, 각자의 분할 seed를 별도로 설정하며 각 산업은 두 학습기에서 각각 자기 검증 Fold의 학습에서 제외됩니다. 트리 수·깊이·학습률·최소 가중치·행/열 추출률·정규화와 학습 seed도 GPS와 Mu에서 각각 선택했습니다. 상세 값은 실행 전 계획과 설정 CSV에 있습니다.",
                numeric_description_ko="bandwidth 배율 0.50, trim 0.01, 유한차분 간격 0.0005, 2022년 기준 통제, 산업 단위 Fold·군집 추론, psmethod=1 공통 GPS Fold 분산 정규밀도, 밀도 하한 없음·ESS 진단만 사용·내부 누락 보간을 유지했습니다. GPS와 Mu는 각자 산업 단위 OOF 예측을 결합합니다. 같은 Fold를 쓰는 기본 동작은 보존하되 독립 Fold는 사용자 요청에 따른 확장입니다.",
                correction_note_ko="이번 200개×20개=4,000개, 논문 방식 누적 1,700개×20개=34,000개 검정으로 보정합니다. 변수·설정을 결과 기준으로 탐색하므로 보정 전 p값이나 이 보정만으로 확증적 결론을 내릴 수 없습니다. 독립 Fold에서도 각 산업의 자기 학습 제외를 보장하며, 기존 R 기본의 공통 Fold와 완전히 동일한 구현이라고 주장하지 않습니다.")
    write_json(dest / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        print(f"Frozen {len(make_plan(args.out)['entries'])} XGBoost specifications")
    else:
        plan = json.loads((args.out / "plan.json").read_text())
        assert plan["planner_sha256"] == file_hash(__file__)
        runner.search(args.out, args.workers)
