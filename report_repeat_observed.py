"""Report all 50 observed-dose repetitions, fixed targets, median inference and stability."""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.pipeline import ROOT
from ps1_observed import write_report
from repeat_observed import DEST
from report_paper_search import PAGE_BREAK, markdown_table, report_plot
from search_paper import evaluate


def checkpoint(dest=DEST):
    dest = Path(dest)
    progress = json.loads((dest / "progress.json").read_text())
    path = dest / f"checkpoint_{len(progress['completed']):02d}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["plan.json", "progress.json", "seeds.csv", "sample.json", "common_targets.json", "common_coverage.csv"]:
            if (dest / name).exists():
                z.write(dest / name, name)
        for p in sorted((dest / "preflight").glob("*.json")):
            z.write(p, p.relative_to(dest))
        for rep in sorted(progress["completed"]):
            for p in sorted((dest / "repetitions" / f"rep_{rep:02d}").rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(dest))
    return path


def describe_stability(table):
    rows = []
    for (year, effect), t in table.groupby(["year", "effect"], sort=False):
        rows.append(dict(year=year, effect=effect, repetitions=len(t), median=float(t.estimate.median()),
                         mean=float(t.estimate.mean()), sd=float(t.estimate.std(ddof=1)), minimum=float(t.estimate.min()),
                         maximum=float(t.estimate.max()), q025=float(t.estimate.quantile(.025)), q975=float(t.estimate.quantile(.975)),
                         positive=int(t.estimate.gt(0).sum()), negative=int(t.estimate.lt(0).sum()),
                         significant_positive=int((t.estimate.gt(0)&t.p_value.lt(.05)).sum()),
                         significant_negative=int((t.estimate.lt(0)&t.p_value.lt(.05)).sum())))
    return pd.DataFrame(rows)


def render(dest=DEST, exports=None, documents=True):
    dest = Path(dest)
    exports = Path(exports or ROOT / "audit/exports/ps1_observed_repeat50")
    exports.mkdir(parents=True, exist_ok=True)
    plan = json.loads((dest / "plan.json").read_text())
    progress = json.loads((dest / "progress.json").read_text())
    count = plan["count"]
    assert progress["status"] == "completed" and set(progress["completed"]) == set(range(count))
    repeats = pd.read_csv(dest / "repeat_estimates.csv")
    common = repeats[repeats.target.eq("common")]
    combined = pd.read_csv(dest / "combined_estimates.csv")
    median = combined[combined.target.eq("common")]
    baseline = common[common.repetition.eq(0)]
    stability = describe_stability(common)
    stability.to_csv(exports / "stability.csv", index=False)
    flags = pd.read_csv(dest / "repeat_flags.csv")
    focused = []
    for r, t in common.groupby("repetition"):
        f = evaluate(t)
        m = t[t.year.eq(2025)&t.effect.eq("moderation")].iloc[0]
        focused.append(dict(repetition=int(r), basic_gate=bool(f["pre_all_nonsignificant"] and f["initial_significant_decline"]),
                            focused=bool(f["pre_all_nonsignificant"] and f["initial_significant_decline"] and f["sme_2025_estimate"]>0 and m.estimate<0 and m.p_value<.05),
                            sme_positive=bool(f["sme_2025_estimate"]>0), sme_sig_positive=f["sme_2025_significantly_positive"],
                            moderation_2025_negative_significant=bool(m.estimate<0 and m.p_value<.05)))
    counts = {name: sum(row[name] for row in focused) for name in list(focused[0])[1:]}
    focused_ids = [row["repetition"] for row in focused if row["focused"]]
    pd.DataFrame(focused).to_csv(exports / "repeat_condition_counts.csv", index=False)
    median_flags = evaluate(median)
    coverage = pd.read_csv(dest / "common_coverage.csv")
    coverage_display = coverage.copy()
    coverage_display["coverage"] = coverage_display.coverage.map(lambda v: f"{v:.2%}")
    summary = dict(count=count, years=plan["config"]["years"], completed_year_runs=count*len(plan["config"]["years"]),
                   target="50-repeat common computational support, by scope; firm-weighted", common_coverage=coverage.to_dict("records"),
                   median_flags=median_flags, repeat_condition_counts=counts,
                   repeat_full_pattern_matches=int(flags.pattern_match.sum()),
                   focused_repetitions=focused_ids,
                   median_estimates=median.replace({np.nan: None}).to_dict("records"),
                   se_formula=plan["se_formula"], seed_changes=["GPS fold", "Mu fold", "GPS learner", "Mu learner"],
                   plan_sha256=file_hash(dest / "plan.json"), base_source_sha256=plan["base_source_sha256"], runner_sha256=plan["runner_sha256"],
                   data_sha256=plan["data_sha256"], os=plan["os"], versions=plan["versions"])
    write_json(exports / "summary.json", summary)
    c = PaperConfig(**{**plan["config"], "label": "ps1_observed_median_of_50"})
    write_report(median, json.loads((dest / "sample.json").read_text()), c, dest)
    html_path = dest / "report.html"
    text = html_path.read_text(encoding="utf-8")
    text = text.replace("<h1>", "<p>50회 공통 평균 대상·추정치 중앙값입니다. Fold 및 GPS/Mu 학습 seed를 모두 변경했고, SE에는 반복 변동을 반영했습니다. 아래 설정은 기준 설정이며 실제 seed 목록은 seeds.csv에 있습니다.</p><h1>", 1)
    html_path.write_text(text, encoding="utf-8")
    def value(year, effect):
        return median[median.year.eq(year)&median.effect.eq(effect)].iloc[0]
    initial, sme, moderation = value(2023, "total"), value(2025, "sme"), value(2025, "moderation")
    lines = ["# PS1 관측점 직접 계산: 1538번 기반 50회 안정성 실험", "",
             f"{count}회×{len(plan['config']['years'])}개 연도={summary['completed_year_runs']}개 연도 실행을 모두 완료했습니다. 기본 1538번에서 GPS/Mu 산업 Fold seed와 두 XGBoost의 학습 random_state만 변경했습니다. 첫 반복은 원래 1538번입니다. 유의성·부호로 반복을 선별하거나 중단하지 않았습니다.", "",
             f"50회 중앙값 기준: 2023년 전체 {initial.estimate:.5f} (p={initial.p_value:.4f}), 2025년 SME {sme.estimate:.5f} (p={sme.p_value:.4f}), 2025년 Large−SME {moderation.estimate:.5f} (p={moderation.p_value:.4f}). 전체 가설 충족은 {median_flags['pattern_match']}입니다. p값은 보정 전 점별 값입니다.", "",
             "## 실험 설계", "",
             "PS1의 D 예측 후 정규밀도, 관측된 처치점에서 ATET(D,D−0.0005)/0.0005 계산, bandwidth 배율 0.5·√5 커널·trim 0.01, GPS 4-Fold·Mu 3-Fold, 동일한 산업 변수와 Mu 기업 변수, 2022년 통제·매출 규모 분류를 유지했습니다. 50회 반복에서는 Fold seed뿐 아니라 열·행 추출에 영향을 주는 두 학습 seed도 함께 바뀝니다. 이전의 격자·Fold만 변경한 50회 실험과는 설계가 다릅니다.", "",
             "Windows·Linux의 XGBoost 열 추출 차이를 피하기 위해, 원래 Linux 결과가 재현되는 고정 컨테이너에서 실행했습니다. Python·패키지·이미지 환경을 유지했으며 4개 seed 채널은 실행 전에 고정한 plan.json/seeds.csv에 있습니다. 모델 파라미터나 변수를 탐색하지 않았습니다.", "",
             "## 평균 대상 고정", "",
             "결과변수를 사용하지 않는 GPS·커널·트리밍 사전 점검으로 반복별 계산 가능 관측점을 구한 뒤, 50회 모두에서 가능한 점의 교집합을 정했습니다. 본문 중앙값과 빈도는 이 공통 평균 대상을 사용합니다. 각 처치점의 효과 곡선은 원래 전체 기업에서 추정하고, 평균 가중치의 대상만 고정합니다. 산업별 같은 가중치가 아니라 기업 수 비중을 사용합니다.", "",
             markdown_table(coverage_display.rename(columns=dict(scope="집단", n_target_firms="원래 기업 수", n_used_firms="공통 포함 기업 수", coverage="포함 비율", dose_points="관측점 수"))), "",
             "원래 모든 기업에 대한 효과와 공통 계산 가능 대상의 평균은 구분해야 합니다. 각 반복 자체의 가능한 대상(native)으로 계산한 결과도 별도 CSV에 보존합니다. 공통 대상에서 필요한 점이 실패하면 해당 반복을 조용히 버리지 않고 실험을 실패로 처리하도록 검증했습니다. ESS 제외·밀도 하한·관측점 보간은 사용하지 않았습니다.", "",
             "## 중앙값과 표준오차", "",
             "효과별 추정치는 50회 중앙값입니다. 표준오차는 sqrt{median[SE_r²+(추정치_r−중앙값)²]}로 분할·학습에 따른 변동을 반영하고, p값과 95% 구간은 이 중앙값과 표준오차로 다시 계산했습니다. 50개 p값의 중앙값을 사용하지 않았고 SE를 √50으로 나누지도 않았습니다.", "",
             "각 반복에서 Large−SME를 먼저 계산한 뒤 그 조절효과를 중앙값 집계합니다. 따라서 최종 조절효과 중앙값은 두 규모 집단의 중앙값 차이와 다를 수 있습니다. 같은 표본을 재학습한 50회이며 독립 표본 50개가 아닙니다. 아래 반복별 빈도와 2.5~97.5% 범위는 계산 안정성 진단이며 인과효과의 신뢰구간과 구분합니다.", ""]
    for name, t in [("50회 중앙값", median), ("원래 seed: 반복 0, 같은 공통 대상", baseline)]:
        folder = exports / ("median" if name == "50회 중앙값" else "baseline")
        folder.mkdir(exist_ok=True)
        report_plot(t, folder, {"label": "PS1 observed | " + ("median of 50" if name == "50회 중앙값" else "original seed")}, height=5.)
        display = []
        for year in [2020, 2021, 2023, 2024, 2025]:
            row = {"연도": year}
            for effect, title in [("total", "전체 (p)"), ("sme", "SME (p)"), ("large", "Large (p)"), ("moderation", "Large−SME (p)")]:
                r = t[t.year.eq(year)&t.effect.eq(effect)].iloc[0]
                row[title] = f"{r.estimate:.5f} ({r.p_value:.4f})"
            display.append(row)
        lines += [PAGE_BREAK, "## " + name, "", markdown_table(pd.DataFrame(display)), "",
                  f"![{name}]({folder.relative_to(ROOT).as_posix()}/report_ACRT.png)", ""]
    condition_names = dict(basic_gate="사전 비유의 + 2023 전체 유의한 하락", focused="기본 기준 + SME2025 양수 + 조절2025 음수 유의",
                           sme_positive="SME2025 양수", sme_sig_positive="SME2025 유의한 양수",
                           moderation_2025_negative_significant="조절2025 음수 유의")
    lines += [PAGE_BREAK, "## 반복별 조건 충족 빈도", "",
              markdown_table(pd.DataFrame([dict(조건=condition_names[k], 충족반복수=v, 전체반복수=count) for k,v in counts.items()])), "",
              f"동시 조건(focused)을 충족한 반복 번호: {focused_ids}. 반복 0은 실행 전에 포함한 원래 1538번 seed입니다. 이 빈도를 독립 실험의 성공 확률로 해석하지 않습니다.", "",
              f"기존 전체 가설 패턴을 모두 충족한 반복 수: {summary['repeat_full_pattern_matches']}/{count}. 성공한 반복을 골라 최종 추정치로 쓰지 않았습니다.", "",
              "## 반복별 분포", "", "2.5~97.5%는 50개 점추정치의 분포 범위이며 95% 인과 신뢰구간이 아닙니다.", ""]
    display = stability[["year", "effect", "median", "q025", "q975", "positive", "significant_positive", "significant_negative"]].copy()
    for col in ["median", "q025", "q975"]:
        display[col] = display[col].map(lambda v: f"{v:.5f}")
    display = display.rename(columns=dict(year="연도", effect="효과", median="중앙값", q025="2.5%", q975="97.5%", positive="양수 횟수", significant_positive="양수 유의", significant_negative="음수 유의"))
    lines += [markdown_table(display), "", "## 설정·해석·파일", "",
              "모든 반복의 변수 목록과 모델 파라미터는 plan.json 및 repetitions/rep_XX/config.json에 있습니다. 초기 설정의 GPS·Mu 모델 파라미터는 다음과 같으며 random_state만 반복별로 바뀝니다.", "",
              "GPS: " + json.dumps(plan["config"]["gps_params"]) + ".", "",
              "Mu: " + json.dumps(plan["config"]["mu_params"]) + ".", "",
              "기업 변수: " + ", ".join(plan["config"]["mu_firm_features"]) + ".", "",
              "산업 변수: " + ", ".join(plan["config"]["gps_features"]) + ". 두 학습기에 동일 산업 더미를 추가했습니다.", "",
              "반복으로 안정성을 확인하더라도 식별·평활 편향·산업 군집 및 ACRT 확장에 대한 이론적 조건이 자동 검증되지는 않습니다. 과거 탐색으로 선택한 1538번의 후속 안정성 분석이므로 보정 전 p값은 탐색적으로 해석해야 합니다.", "",
              "repeat_estimates.csv/combined_estimates.csv: 반복별 결과와 중앙값 집계. stability.csv: 분포·부호·유의성 빈도. repeat_flags.csv/repeat_condition_counts.csv: 모든 반복의 조건 판정. repetitions/ 아래에는 연도별 계산 진단과 산업 점수가 있습니다. 학습 모델·원자료는 업로드에 포함하지 않았습니다.", "",
              f"계획 SHA256: `{summary['plan_sha256']}`", "", f"실행기 SHA256: `{summary['runner_sha256']}`", ""]
    md = exports / "PS1_Observed_Repeat50_ko.md"
    md.write_text("\n".join(lines))
    for ext, extra in ([("pdf", ["--pdf-engine=xelatex", "-V", "mainfont=Noto Sans CJK KR", "-V", "sansfont=Noto Sans CJK KR",
                                "-V", "geometry:margin=20mm", "-V", "fontsize=10pt", "-V", "papersize=a4",
                                "-V", r"header-includes=\setlength{\emergencystretch}{3em}\sloppy"]), ("docx", [])] if documents else []):
        subprocess.run(["pandoc", str(md), *extra, "-o", str(exports / f"PS1_Observed_Repeat50_ko.{ext}")], cwd=ROOT, check=True)
    for name in ["plan.json", "progress.json", "sample.json", "seeds.csv", "common_targets.json", "common_coverage.csv", "repeat_estimates.csv", "combined_estimates.csv", "repeat_flags.csv"]:
        shutil.copy2(dest / name, exports / name)
    archive = exports / "PS1_Observed_Repeat50_Results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file() and ".scratch" not in p.parts and not p.name.startswith("checkpoint_"):
                z.write(p, p.relative_to(dest))
        for p in sorted(exports.rglob("*")):
            if p.is_file() and p.name in ["stability.csv", "repeat_condition_counts.csv", "summary.json", "report_ACRT.png"]:
                z.write(p, p.relative_to(exports))
        z.writestr(md.name, md.read_text().replace(exports.relative_to(ROOT).as_posix()+"/", ""))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["checkpoint", "report", "summary"])
    p.add_argument("--out", type=Path, default=DEST)
    args = p.parse_args()
    print(checkpoint(args.out) if args.command == "checkpoint" else json.dumps(render(args.out, documents=args.command == "report"), ensure_ascii=False))
