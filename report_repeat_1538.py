"""Korean PDF/Word comparison of the three predefined 1538 sensitivity cases."""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from labor_did.io import file_hash, write_json
from paper_did.pipeline import ROOT
from repeat_paper_1538 import DEST
from report_paper_search import (
    PAGE_BREAK,
    markdown_table,
    report_plot,
    size_description,
)
from search_paper import evaluate, primary

NAMES = {"diff_1e5": "유한차분 간격만 1e-5", "repeat_50": "Fold 반복만 50회", "both": "간격 1e-5 + Fold 50회"}
REASONS = {"initial_2023": "2023년 전체 유의한 하락 미충족", "pre_2020": "2020년 사전 기준 미충족",
           "pre_2021": "2021년 사전 기준 미충족", "numerical_failure": "수치 계산 실패"}


def checkpoint(dest=DEST):
    dest = Path(dest)
    progress = json.loads((dest / "progress.json").read_text())
    last = sum(progress["completed_repetitions"].values())
    path = dest / f"checkpoint_{last:03d}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["plan.json", "progress.json"]:
            z.write(dest / name, name)
        # Prior gates must all pass before a case's estimates are published.
        ready = all(y in progress["completed_years"] for y in [2023, 2020, 2021])
        if ready:
            for folder in (dest / "retained").iterdir():
                for file in folder.rglob("*"):
                    if file.is_file():
                        z.write(file, file.relative_to(dest))
    return path


def render(dest=DEST, exports=None):
    dest = Path(dest)
    exports = Path(exports or ROOT / "audit/exports/paper_1538_sensitivity")
    exports.mkdir(parents=True, exist_ok=True)
    plan = json.loads((dest / "plan.json").read_text())
    progress = json.loads((dest / "progress.json").read_text())
    assert progress["status"] == "completed" and len(progress["cases"]) == 3
    parent = ROOT / plan["parent_path"]
    baseline = exports / "baseline"
    baseline.mkdir(exist_ok=True)
    for name in ["config.json", "estimates.csv", "sample.json"]:
        shutil.copy2(parent / name, baseline / name)
    folders = [("baseline", "원래 1538", baseline, 1)]
    rows, diagnostics = [], []
    for entry in plan["entries"]:
        label = entry["label"]
        state = progress["cases"][label]
        row = dict(case=label, description=NAMES[label], diff_step=entry["config"]["diff_step"],
                   repetitions=entry["repetitions"], status=state["status"], reason=state.get("reason", ""))
        if state["status"] == "completed":
            folder = dest / "retained" / label
            t = primary(pd.read_csv(folder / "estimates.csv"))
            flags = evaluate(t)
            row.update(flags)
            for year, effect in [(2023, "total"), (2025, "sme"), (2025, "moderation")]:
                r = t[t.year.eq(year) & t.effect.eq(effect)].iloc[0]
                row[f"{effect}_{year}"] = r.estimate
                row[f"{effect}_{year}_p"] = r.p_value
            folders.append((label, NAMES[label], folder, entry["repetitions"]))
            diagnostics.append(t.assign(case=label))
        rows.append(row)
    pd.DataFrame(rows).to_csv(exports / "comparison.csv", index=False)
    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(exports / "combined_estimates.csv", index=False)
    else:
        pd.DataFrame(columns=["case", "year", "effect", "estimate", "se", "p_value"]).to_csv(exports / "combined_estimates.csv", index=False)
    labels = [r["case"] for r in rows if r["status"] == "completed"]
    summary = dict(planned=3, completed_cases=labels, rejected_cases={r["case"]: r["reason"] for r in rows if r["status"] == "rejected"},
                   planned_repetitions=[1, 50, 50], aggregation=plan["aggregation"], parent="paper_search_1538",
                   plan_sha256=file_hash(dest / "plan.json"), source_sha256=plan["source_sha256"], runner_sha256=plan["runner_sha256"],
                   data_sha256=plan["data_sha256"], full_pattern_matches=sum(bool(r.get("pattern_match")) for r in rows))
    write_json(exports / "summary.json", summary)
    lines = ["# 1538번 유한차분·반복 Fold 민감도 비교", "",
             "1538번의 변수·XGBoost 파라미터·bandwidth 0.5·trim 0.01·매출 규모 기준을 유지하고, 사전에 정한 세 가지 변경만 시험했습니다. 유한차분 간격 1e-5는 기존 0.0005의 1/50입니다.", "",
             "## 비교 설정과 저장 상태", "",
             markdown_table(pd.DataFrame([dict(설정=r["description"], 차분간격=r["diff_step"], 반복수=r["repetitions"],
                                              상태="기준 통과·저장" if r["status"] == "completed" else REASONS[r["reason"]]) for r in rows])), "",
             "GPS 4-Fold와 Mu 3-Fold를 유지하며 각자의 산업 분할 seed만 50회 변경했습니다. 첫 반복은 원래 GPS seed 107·Mu seed 73입니다. 각 XGBoost의 random_state는 원래 값으로 고정했습니다. 같은 50개 분할을 두 반복 설정에서 공유해 간격 변경을 비교합니다.", "",
             "## 중앙값과 추론", "",
             "각 반복에서 전체·SME·Large와 Large−SME를 먼저 계산하고, 효과별 50회 추정치의 중앙값을 사용합니다. 표준오차는 sqrt{median[SE_r² + (추정치_r − 중앙값)²]}입니다. 동일한 기업 표본을 재분할한 것이므로 SE를 √50으로 나누지 않습니다. 조절효과 중앙값은 각 반복의 Large−SME 차이를 집계한 것이어서, 최종 두 집단 중앙값의 단순 차이와 다를 수 있습니다.", "",
             "50회 반복의 분할 안정화를 적용한 민감도 분석이며, 이 산업 군집·ACRT 확장에 저자의 원래 추론 정리가 그대로 적용된다고 주장하지 않습니다. 보조 bootstrap 구간은 반복 중앙값에 재사용하지 않았습니다. 보고서의 p값과 구간은 보정 전 점별 값입니다. 새 3개 설정×20개=60개 보정과 과거 1,700개를 포함한 34,060개 보정은 결과 CSV에 별도로 기록했습니다. 50회 반복은 50개의 독립 표본이나 추가 가설 선택으로 취급하지 않습니다.", "",
             "## 중단·보존 규칙", "",
             "2023년 전체 음수·p<0.05, 2020·2021년 모든 네 효과 p≥0.05를 요구합니다. 반복 설정에서는 50회를 모두 결합한 추정치로 판정하고, 유리한 반복만 선택하지 않습니다. 어느 기준이라도 실패하면 이후 연도 계산을 중단하고 그 설정의 추정치·그래프·반복별 결과를 삭제했습니다. 위 상태표에는 탈락 유형만 표시했습니다.", "",
             "간격 1e-5는 현재 구현에서 차분 폭과 평가 격자 폭에 동시에 적용됩니다. 모형·bandwidth·커널 √5는 유지했습니다. XGBoost의 예측은 계단 형태이므로 간격이 더 작다는 이유만으로 더 정확한 추정이라고 해석하지 않습니다.", ""]
    for label, title, folder, repeats in folders:
        config = json.loads((folder / "config.json").read_text())
        sample = json.loads((folder / "sample.json").read_text())
        t = primary(pd.read_csv(folder / "estimates.csv"))
        report_plot(t, folder, config, height=5.)
        relative = folder.relative_to(ROOT).as_posix() if label != "baseline" else "baseline"
        lines += [PAGE_BREAK, f"## {title}", "", f"유한차분 간격 {config['diff_step']}; 반복 {repeats}회; GPS 4-Fold·Mu 3-Fold.", "",
                  size_description(config, sample.get("size_classification")), ""]
        display = []
        for year in [2020, 2021, 2023, 2024, 2025]:
            record = {"연도": year}
            for effect, heading in [("total", "전체 (p)"), ("sme", "SME (p)"), ("large", "Large (p)"), ("moderation", "Large−SME (p)")]:
                r = t[t.year.eq(year) & t.effect.eq(effect)].iloc[0]
                record[heading] = f"{r.estimate:.5f} ({r.p_value:.4f})"
            display.append(record)
        lines += [markdown_table(pd.DataFrame(display)), "", f"![{title}]({relative}/report_ACRT.png)", ""]
        if label != "baseline":
            m = t[t.year.eq(2025) & t.effect.eq("moderation")].iloc[0]
            s = t[t.year.eq(2025) & t.effect.eq("sme")].iloc[0]
            lines += [f"2025년 SME 양수: {bool(s.estimate > 0)}; SME 유의한 양수: {bool(s.estimate > 0 and s.p_value < .05)}; Large−SME 음수·유의: {bool(m.estimate < 0 and m.p_value < .05)}. 최소 반복 coverage {t.coverage.min():.2%}.", ""]
    config = json.loads((parent / "config.json").read_text())
    lines += [PAGE_BREAK, "## 유지한 변수·모델 설정", "",
              "GPS 및 Mu 산업 변수: " + ", ".join(config["gps_features"]) + ". 산업 더미는 동일하게 추가합니다.", "",
              "Mu 기업 변수: " + ", ".join(config["mu_firm_features"]) + ". Mu에는 처치 D를 별도 추가합니다.", "",
              "GPS XGBoost 파라미터: " + json.dumps(config["gps_params"]) + ".", "",
              "Mu XGBoost 파라미터: " + json.dumps(config["mu_params"]) + ".", "",
              "스플라인 변환 없음; 2022년 기준 통제; 산업 GPS·산업 Fold·산업 군집 표준오차; psmethod=1 공통 GPS 검증 Fold 분산; ESS 제외 없음; 밀도 하한 없음; 내부 누락 보간. 반복별 상세 seed·추정치·군집 점수와 audit는 통과 설정의 결과 ZIP에 있습니다.", "",
              f"계획 SHA256: `{summary['plan_sha256']}`", "", f"추정 소스 SHA256: `{summary['source_sha256']}`", ""]
    md = exports / "Paper_1538_Sensitivity_ko.md"
    # Images outside the export folder resolve relative to the repository for pandoc.
    text = "\n".join(lines).replace("](baseline/", "](" + (baseline.relative_to(ROOT)).as_posix() + "/")
    md.write_text(text)
    for ext, extra in [("pdf", ["--pdf-engine=xelatex", "-V", "mainfont=Noto Sans CJK KR", "-V", "sansfont=Noto Sans CJK KR",
                                "-V", "geometry:margin=20mm", "-V", "fontsize=10pt", "-V", "papersize=a4",
                                "-V", r"header-includes=\setlength{\emergencystretch}{3em}\sloppy"]), ("docx", [])]:
        subprocess.run(["pandoc", str(md), *extra, "-o", str(exports / f"Paper_1538_Sensitivity_ko.{ext}")], cwd=ROOT, check=True)
    for name in ["plan.json", "progress.json"]:
        shutil.copy2(dest / name, exports / name)
    archive = exports / "Paper_1538_Sensitivity_Results.zip"
    portable = text.replace("results/paper_1538_sensitivity/retained/", "retained/").replace(baseline.relative_to(ROOT).as_posix() + "/", "baseline/")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in exports.rglob("*"):
            if p.is_file() and p != archive and p != md and p.suffix not in [".pdf", ".docx"]:
                z.write(p, p.relative_to(exports))
        z.writestr(md.name, portable)
        for p in sorted((dest / "retained").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(dest))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["checkpoint", "report"])
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    print(checkpoint(args.out) if args.command == "checkpoint" else json.dumps(render(args.out), ensure_ascii=False))
