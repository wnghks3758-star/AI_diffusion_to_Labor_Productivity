"""Change only the size variable in 200 previously gate-passed paper specifications."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import search_paper as engine
import search_paper_batch as runner
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.models import source_hash
from paper_did.pipeline import ROOT
from report_focused_history import selection
from search_paper_more import effective_fingerprint
from search_paper_size import PRIOR as FIRST_BATCHES

DEST = ROOT / "audit/results/paper_search_1301_1500"
PRIOR = [*FIRST_BATCHES, ROOT / "audit/results/paper_search_1101_1300"]
COUNT = 200


def load_parents(prior):
    candidates = []
    for path in map(Path, prior):
        plan = json.loads((path / "plan.json").read_text())
        for config_file in sorted((path / "retained").glob("*/config.json")):
            estimate_file = config_file.with_name("estimates.csv")
            flags = selection(pd.read_csv(estimate_file))
            if flags is None or not flags["gate_pass"]:
                continue
            c = PaperConfig(**json.loads(config_file.read_text()))
            if c.size_variable != "sale":
                continue
            candidates.append(dict(number=int(c.label.rsplit("_", 1)[-1]), config=c.to_dict(),
                                   config_sha256=file_hash(config_file), estimates_sha256=file_hash(estimate_file),
                                   source_sha256=plan["source_sha256"],
                                   config_path=str(config_file.relative_to(ROOT) if config_file.is_relative_to(ROOT) else config_file),
                                   estimates_path=str(estimate_file.relative_to(ROOT) if estimate_file.is_relative_to(ROOT) else estimate_file)))
    candidates.sort(key=lambda x: x["number"])
    assert len({effective_fingerprint(p["config"]) for p in candidates}) == len(candidates)
    return candidates


def make_plan(dest=DEST, prior=PRIOR):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    prior = list(map(Path, prior))
    plans = [json.loads((p / "plan.json").read_text()) for p in prior]
    summary = json.loads((prior[-1] / "summary.json").read_text())
    assert summary["cumulative_attempted"] == 1300
    eligible = load_parents(prior)
    if len(eligible) < COUNT:
        raise ValueError(f"Need {COUNT} previously passed parents, found {len(eligible)}")
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, COUNT, dtype=int)]
    seen = {effective_fingerprint(e["config"]) for p in plans for e in p["entries"]}
    families, entries = {}, []
    family_sizes = defaultdict(int)
    for parent in selected:
        number = 1301 + len(entries)
        c = PaperConfig(**{**parent["config"], "size_variable": "assets_annual", "label": f"paper_search_{number}"})
        base, changed = PaperConfig(**parent["config"]).to_dict(), c.to_dict()
        differences = {k for k in base if base[k] != changed[k]}
        assert differences == {"label", "size_variable"}, differences
        key = effective_fingerprint(changed)
        assert key not in seen
        seen.add(key)
        family_key = effective_fingerprint({**changed, "bandwidth_factor": .7, "trim": .1})
        family = families.setdefault(family_key, len(families))
        variant = family_sizes[family]
        family_sizes[family] += 1
        entries.append(dict(number=number, family=family, family_name=f"preserved_fit_{family + 1}", variant=variant,
                            config=changed, fingerprint=c.fingerprint, effective_fingerprint=key,
                            parent_label=base["label"], parent_number=parent["number"],
                            parent_config_path=parent["config_path"], parent_estimates_path=parent["estimates_path"],
                            parent_config_sha256=parent["config_sha256"], parent_estimates_sha256=parent["estimates_sha256"],
                            parent_source_sha256=parent["source_sha256"], changed_fields=["size_variable"]))
    assert len(entries) == COUNT and len({e["parent_label"] for e in entries}) == COUNT
    plan = {k: plans[-1][k] for k in ["hypothesis_estimand", "alpha", "retention", "criteria", "focused_filter"]}
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), count=COUNT, entries=entries, size_search=True,
                eligible_parent_count=len(eligible), selected_parent_count=COUNT, parent_size_variable="sale", size_variable_only=True,
                parent_selection="Sort previously gate-passed configurations by original run number and select 200 evenly spaced ranks; no 2025 outcome ranking.",
                runner_sha256=file_hash(runner.__file__), planner_sha256=file_hash(__file__), engine_sha256=file_hash(engine.__file__),
                source_sha256=source_hash(), runner_file="search_paper_batch.py", planner_file="search_paper_size_from_passed.py",
                data_sha256=file_hash(ROOT / entries[0]["config"]["data_path"]), previous_summary=summary,
                previous_count=1300, cumulative_count=1500, correction_family_size=4000, cumulative_correction_family_size=30000,
                prior_plan_sha256=[file_hash(p / "plan.json") for p in prior],
                previous_summary_sha256=file_hash(prior[-1] / "summary.json"), screening_order=list(engine.SCREEN_ORDER),
                stop_rule="Stop failed 2023/2020/2021 gates immediately and delete rejected results; attempt all 200.",
                report_stem="Paper_Search_1301_1500", report_title_ko="이전 통과 설정 유지: 규모 변수만 변경한 추가 200개",
                design_description_ko=f"이전 논문 방식의 저장 기준 통과 {len(eligible)}개 중 실행 번호 순으로 고르게 200개를 선정했습니다. 각 기준 설정의 모델·변수·수치 설정을 그대로 두고 기업 규모 변수만 sale에서 assets_annual로 변경했습니다. 기준 설정은 과거 결과를 보고 선정된 탐색 결과이며 확증적 검정이 아닙니다.",
                variables_description_ko="GPS·Mu에 포함된 개별 변수, 산업 더미와 기업 규모 더미의 포함 여부를 각 기준 설정 그대로 유지했습니다. 새로운 통제변수 선택이나 학습기 변경은 하지 않았습니다. 원본 설정·추정 결과의 경로와 해시, 변경 필드를 실행 전 계획에 기록했습니다.",
                models_description_ko="학습기 종류·하이퍼파라미터·Fold 수·seed·산업 분류 기준·bandwidth·trim은 각 부모 설정과 동일합니다. 규모 더미를 Mu에 사용하는 경우에는 새 자산 기준 더미로 다시 학습합니다. 규모 더미를 사용하지 않으면 같은 학습 결과를 재사용하되 규모별 추정과 검정은 새 집단으로 다시 계산합니다.",
                numeric_description_ko="유일한 실질 설정 변경은 size_variable: sale → assets_annual입니다. 규모 산정 기간은 기존 2020~2022년, 기업별 요약은 mean, 산업 내 기준은 median, 결측 처리와 소규모 산업 미분류 기준도 기존 설정을 유지했습니다. 분석연도·기준연도·표본·ATET 점수·ACRT 변환은 유지합니다. 변수 이름은 한 개만 바뀌지만 기업 규모 더미와 SME/Large 집단의 구성은 바뀝니다.",
                correction_note_ko="이번 200개×20개=4,000개, 논문 방식 누적 1,500개×20개=30,000개 검정으로 보정합니다. 과거 통과 결과를 바탕으로 수행하는 후속 탐색이며, 규모 정의 변경으로 집단 구성과 추정 대상도 달라집니다. 보정 전 p값이나 이 보정만으로 확증적 결론을 내리지 않습니다.")
    write_json(dest / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        print(f"Frozen {len(make_plan(args.out)['entries'])} size-variable-only configurations")
    else:
        plan = json.loads((args.out / "plan.json").read_text())
        assert plan["planner_sha256"] == file_hash(__file__)
        runner.search(args.out, args.workers)
