"""Freeze and run 200 additional paper-aligned specifications, 601--800."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import search_paper as engine
import search_paper_more as runner
from labor_did.config import INDUSTRY_CONTROLS
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT

DEST = ROOT / "audit/results/paper_search_601_800"
PRIOR = [
    ROOT / "audit/results" / name
    for name in ["paper_search_200", "paper_search_201_400", "paper_search_401_600"]
]
SEED = 2026102306
VARIANTS = [(0.50, 0.05), (0.60, 0.075), (0.70, 0.10), (0.90, 0.075), (1.20, 0.15)]


def make_plan(dest=DEST, prior=PRIOR):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    plans = [json.loads((Path(p) / "plan.json").read_text()) for p in prior]
    summary = json.loads((Path(prior[-1]) / "summary.json").read_text())
    assert summary["cumulative_attempted"] == 600
    old = {e["number"]: e for p in plans for e in p["entries"]}
    assert len(old) == 600
    seen = {runner.effective_fingerprint(e["config"]) for e in old.values()}
    rng = np.random.default_rng(SEED)
    families, family_keys = [], set()
    firm_pool = PaperConfig().mu_firm_features
    for number in [402, 407, 432, 447]:
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
                c = runner.toggle(center, field, variable, INDUSTRY_CONTROLS)
            key = runner.effective_fingerprint(
                {**c, "bandwidth_factor": 0.7, "trim": 0.1}
            )
            if key in family_keys:
                continue
            if any(
                runner.effective_fingerprint({**c, "bandwidth_factor": b, "trim": t})
                in seen
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
    for i in range(20):
        c = PaperConfig().to_dict()
        for field, pool in [
            ("gps_features", INDUSTRY_CONTROLS),
            ("mu_industry_features", INDUSTRY_CONTROLS),
            ("mu_firm_features", firm_pool),
        ]:
            c[field] = [
                v for v in pool if rng.random() < [0.35, 0.60, 0.85][i % 3]
            ] or [pool[0]]
        gps, mu = models[i % 6], models[(i * 3 + i // 6 + 1) % 6]

        def params(model):
            if model == "random_forest":
                return dict(
                    n_estimators=150,
                    max_depth=[2, 3, 5][i % 3],
                    min_samples_leaf=[5, 10, 15][i % 3],
                )
            if model == "xgboost":
                return dict(
                    n_estimators=[150, 250][i % 2],
                    max_depth=[2, 3][i % 2],
                    learning_rate=[0.025, 0.05][i % 2],
                    min_child_weight=[8, 12][i % 2],
                )
            if model == "lasso":
                return dict(alpha=[0.0003, 0.003, 0.03][i % 3])
            if model == "ridge":
                return dict(alpha=[0.05, 0.5, 5.0, 50.0][i % 4])
            return {}

        c.update(
            gps_model=gps,
            mu_model=mu,
            gps_params=params(gps),
            mu_params=params(mu),
            n_splits=[3, 4, 5][i % 3],
            seed=[19, 101, 2028, 311][i % 4],
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
            number = 601 + len(entries)
            c = PaperConfig(
                **{
                    **base,
                    "bandwidth_factor": bw,
                    "trim": trim,
                    "label": f"paper_search_{number}",
                }
            )
            key = runner.effective_fingerprint(c.to_dict())
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
    assert len(entries) == 200
    plan = {
        k: plans[-1][k]
        for k in ["hypothesis_estimand", "alpha", "retention", "criteria"]
    }
    plan.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        count=200,
        seed=SEED,
        entries=entries,
        runner_sha256=file_hash(runner.__file__),
        planner_sha256=file_hash(__file__),
        engine_sha256=file_hash(engine.__file__),
        source_sha256=source_hash(),
        runner_file="search_paper_more.py",
        planner_file="search_paper_fourth.py",
        data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]),
        previous_plan_sha256=file_hash(Path(prior[-1]) / "plan.json"),
        previous_summary_sha256=file_hash(Path(prior[-1]) / "summary.json"),
        prior_plan_sha256=[file_hash(Path(p) / "plan.json") for p in prior],
        previous_summary=summary,
        previous_count=600,
        cumulative_count=800,
        correction_family_size=4000,
        cumulative_correction_family_size=16000,
        screening_order=list(engine.SCREEN_ORDER),
        stop_rule="Stop failed 2023/2020/2021 gates immediately; attempt all 200.",
        focused_filter={
            "sme_year": 2025,
            "sme_sign": "positive",
            "moderation_year": 2025,
            "moderation_sign": "negative",
            "alpha": 0.05,
        },
        report_stem="Paper_Search_601_800",
        report_title_ko="논문 방식 추가 탐색: 601~800번",
        design_description_ko="이전 600개와 중복 없는 200개를 실행 전에 확정했습니다. SME 2025가 양수였던 402·407·432·447번 주변 20개 변수 조합과 새로운 조합 20개를 각 5개 수치 설정으로 비교합니다. 기존 양수 추정치는 유의하지 않았으며, 결과를 보고 정한 후속 탐색입니다.",
        variables_description_ko="GPS/Mu 산업 변수를 개별 추가·제외하거나 기업 통제변수를 개별 선택합니다. 새로운 조합은 각 변수 포함 확률 0.35/0.60/0.85로 추출합니다. seed 2026102306입니다.",
        models_description_ko="GPS·Mu의 Spline/RF/Lasso/Ridge/XGBoost/선형, Fold 3/4/5, seed·산업 더미·최소 관측수·모형 설정을 변경합니다. ESS는 진단값이며 제외 기준은 없습니다.",
        numeric_description_ko="(bandwidth 배율, trim): (0.50,0.05), (0.60,0.075), (0.70,0.10), (0.90,0.075), (1.20,0.15). 추정식·기준연도·표본·산업 군집 추론은 유지합니다.",
        correction_note_ko="배치 보정은 4,000개, 논문 방식 800개 누적 보정은 16,000개 검정입니다. 중단 설정도 포함합니다. 이전 방식 탐색은 포함하지 않으며, 적응적 탐색 결과이므로 확증적 검정으로 해석하지 않습니다.",
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
