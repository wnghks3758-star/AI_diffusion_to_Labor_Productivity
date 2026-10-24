"""Report and safe annual checkpoints for observed-dose 1538 comparisons."""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from compare_1538_observed import DEST
from labor_did.io import file_hash, write_json
from paper_did.pipeline import ROOT
from report_paper_search import PAGE_BREAK, markdown_table, report_plot
from search_paper import evaluate

NAMES = {"ps1_grid": "PS1·XGBoost·격자 보간", "ps1_observed": "PS1·XGBoost·관측점 직접", "ps2_rf_observed": "PS2·Random Forest·관측점 직접"}
TARGETS = {"native_supported": "각 방법의 계산 가능 대상", "full_observed_strict": "전체 기업·실패 전파",
           "common_ps1": "PS1 두 방식 공통 대상", "common_all": "세 방식 공통 대상"}


def checkpoint(dest=DEST):
    dest = Path(dest)
    progress = json.loads((dest / "progress.json").read_text())
    path = dest / f"checkpoint_{len(progress['completed_years'])}_years.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["plan.json", "progress.json", "sample.json", "kernel_gps_oof.npz"]:
            if (dest / name).exists():
                z.write(dest / name, name)
        for year in progress["completed_years"]:
            for p in sorted((dest / "years" / str(year)).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(dest))
    return path


def differences(table, target, left, right):
    columns = ["estimate", "se", "p_value", "coverage", "n_used_firms"]
    a = table[table.target.eq(target) & table.method.eq(left)].set_index(["year", "effect"])[columns]
    b = table[table.target.eq(target) & table.method.eq(right)].set_index(["year", "effect"])[columns]
    assert a.index.equals(b.index)
    out = a.subtract(b).add_prefix("difference_").reset_index()
    out["left"], out["right"], out["target"] = left, right, target
    return out


def render(dest=DEST, exports=None):
    dest = Path(dest)
    exports = Path(exports or ROOT / "audit/exports/paper_1538_observed")
    exports.mkdir(parents=True, exist_ok=True)
    plan = json.loads((dest / "plan.json").read_text())
    progress = json.loads((dest / "progress.json").read_text())
    assert progress["status"] == "completed" and len(progress["completed_years"]) == 5
    table = pd.read_csv(dest / "estimates.csv")
    ps1diff = differences(table, "common_ps1", "ps1_observed", "ps1_grid")
    all_diff = differences(table, "common_all", "ps2_rf_observed", "ps1_observed")
    pd.concat([ps1diff, all_diff], ignore_index=True).to_csv(exports / "matched_differences.csv", index=False)
    coverage = table[table.year.eq(2023) & table.effect.ne("moderation")][["method", "target", "effect", "n_target_firms", "n_used_firms", "coverage", "status"]]
    coverage.to_csv(exports / "coverage.csv", index=False)
    flags = []
    for (method, target), t in table.groupby(["method", "target"], sort=False):
        flags.append(dict(method=method, target=target, **evaluate(t)))
    pd.DataFrame(flags).to_csv(exports / "pattern_flags.csv", index=False)
    summary = dict(parent="paper_search_1538", methods=list(NAMES), years=plan["years"],
                   diff_step=plan["config"]["diff_step"], repeats=1, gps_fold=4, mu_fold=3,
                   ps1_matched_max_abs_difference=float(ps1diff.difference_estimate.abs().max()),
                   ps2_matched_max_abs_difference=float(all_diff.difference_estimate.abs().max()),
                   all_method_target_2025=table[table.year.eq(2025)].replace({np.nan: None}).to_dict("records"),
                   plan_sha256=file_hash(dest / "plan.json"), core_source_sha256=plan["core_source_sha256"],
                   files=plan["files"], data_sha256=plan["data_sha256"],
                   interpretation="Model/weight/target comparison, no significance screening. PS2 also changes GPS learner; raw p-values are exploratory.")
    write_json(exports / "summary.json", summary)
    rf_flags = evaluate(table[table.method.eq("ps2_rf_observed") & table.target.eq("common_all")])
    lines = ["# 1538번: 격자 보간·관측점 직접 계산과 psmethod=2 비교", "",
             "원래 1538번의 차분 간격 0.0005, 단일 산업 분할(GPS 4-Fold·seed 107 / Mu 3-Fold·seed 73), bandwidth 배율 0.5, trim 0.01, 매출 규모 기준을 유지했습니다. 최근의 간격 1e-5·50회 반복 설정은 사용하지 않았습니다. 2020·2021·2023·2024·2025년을 유의성과 관계없이 모두 계산한 방법 비교입니다.", "",
             f"같은 평균 대상에서 PS1 보간·직접 계산의 20개 추정치 최대 절대 차이는 {summary['ps1_matched_max_abs_difference']:.6f}입니다. "
             f"PS2·RF의 2023년 전체 유의한 하락 기준: {rf_flags['initial_significant_decline']}; 2025년 SME 양수: {bool(rf_flags['sme_2025_estimate'] > 0)}; 전체 가설: {rf_flags['pattern_match']}. 아래 수치는 보정 전 탐색 결과입니다.", "",
             "## 무엇을 비교했는가", "",
             markdown_table(pd.DataFrame([dict(방법=NAMES[m], GPS="산업 D 예측 후 정규밀도" if m != "ps2_rf_observed" else "처치점별 커널 높이 RF 회귀", ACRT="격자에서 기업 D로 보간" if m == "ps1_grid" else "실제 산업 D에서 직접 계산") for m in NAMES])), "",
             "첫 비교는 같은 XGBoost GPS·같은 Mu에서 격자 보간만 직접 계산으로 바꿉니다. 둘째 비교는 GPS를 직접 커널 회귀와 Random Forest로 함께 바꿉니다. 따라서 둘째 차이를 psmethod 하나의 순수한 효과라고 해석하지 않습니다. Mu의 학습 모형·예측은 같은 연도에서 두 GPS 방식에 동일하게 사용했습니다.", "",
             "## 평균 대상과 가중치", "",
             "산업별 효과를 같은 비중으로 평균하지 않고, 각 관측 처치점의 기업 수 / 해당 집단의 평균 대상 기업 수로 가중했습니다. SME와 Large는 각 집단의 기업 수로 별도 평균하고 조절효과는 Large−SME로 계산했습니다. 각 처치점의 ATET 곡선은 모든 원래 기업을 사용해 추정하며, 방법 비교 시 제한하는 것은 곡선을 평균하는 처치 분포입니다.", "",
             "관측점에서 계산이 실패한 경우 직접 결과에 보간·밀도 하한·0 대체를 넣지 않았습니다. 전체 기업을 모두 요구한 full_observed_strict는 필요한 한 점이라도 실패하면 집계를 계산 불가로 표시합니다. native_supported는 각 방법에서 가능한 처치점으로 대상이 제한된 평균이고, common_ps1 및 common_all은 각각 비교 방법들이 공통으로 계산 가능한 같은 평균 대상입니다. 어느 결과도 전체 기업 효과로 바꿔 부르지 않습니다.", "",
             "full_observed_strict의 coverage=1은 요구한 평균 범위를 뜻하며 계산 성공을 뜻하지 않습니다. 실제 성공 여부는 status=failed/completed로 구분합니다. 성공한 제한 대상 평균의 coverage와 함께 확인해야 합니다.", "",
             "원래 격자 결과는 저장된 computability와 동일한 격자를 재구성하고, 같은 nuisance와 가중치로 다시 계산해 원본 1538번의 효과·SE·p값과 1e-9 이내 일치를 확인했습니다. 공통 대상의 격자 결과도 원래 보간 방식으로 다시 가중했습니다.", "",
             "## 계산 가능 범위", ""]
    display = []
    short_names = dict(ps1_grid="PS1 격자", ps1_observed="PS1 직접", ps2_rf_observed="PS2 RF 직접")
    short_targets = dict(native_supported="각 방식 가능", common_ps1="PS1 공통", common_all="세 방식 공통")
    for (method, target), g in coverage[coverage.target.isin(short_targets)].groupby(["method", "target"], sort=False):
        row = {"방법": short_names[method], "평균 대상": short_targets[target]}
        for scope, title in [("total", "전체"), ("sme", "SME"), ("large", "Large")]:
            r = g[g.effect.eq(scope)].iloc[0]
            row[title] = f"{int(r.n_used_firms):,} ({r.coverage:.2%})" if r.status == "completed" else "계산 불가"
        display.append(row)
    sample = json.loads((dest / "sample.json").read_text())
    lines += [f"각 셀은 포함 기업 수(비율)입니다. 분모는 전체 {sample['analysis_firms']:,}개, SME {sample['size_classification']['sme_firms']:,}개, Large {sample['size_classification']['large_firms']:,}개입니다.", "",
              markdown_table(pd.DataFrame(display)), "",
              "위 표는 2023년 기준이며, 연도별 범위가 동일한지도 검증 자료에 기록했습니다. 전체 대상 엄격 집계는 estimates.csv의 full_observed_strict에서 별도 확인할 수 있습니다.", "",
              "## PS1: 보간과 직접 계산의 수치 차이", "",
              f"같은 평균 대상(common_ps1)에서 5개 연도×4개 효과의 추정치 최대 절대 차이는 {summary['ps1_matched_max_abs_difference']:.8g}입니다. 아래 차이는 직접 계산 − 격자 보간이며, 별도의 차이 유의성 검정은 아닙니다.", ""]
    d = ps1diff[["year", "effect", "difference_estimate", "difference_se", "difference_p_value"]].copy()
    for col in d.columns[2:]:
        d[col] = d[col].map(lambda v: f"{v:.7f}")
    lines += [markdown_table(d.rename(columns=dict(year="연도", effect="효과", difference_estimate="추정치 차이", difference_se="SE 차이", difference_p_value="p값 차이"))), ""]
    for target in ["common_ps1", "common_all"]:
        for method in (["ps1_grid", "ps1_observed"] if target == "common_ps1" else list(NAMES)):
            t = table[table.target.eq(target) & table.method.eq(method)]
            folder = exports / "figures" / f"{method}_{target}"
            folder.mkdir(parents=True, exist_ok=True)
            report_plot(t, folder, {"label": f"1538 | {method} | {target}"}, height=5.)
            rows = []
            for year in [2020, 2021, 2023, 2024, 2025]:
                row = {"연도": year}
                for effect, title in [("total", "전체 (p)"), ("sme", "SME (p)"), ("large", "Large (p)"), ("moderation", "Large−SME (p)")]:
                    r = t[t.year.eq(year) & t.effect.eq(effect)].iloc[0]
                    row[title] = f"{r.estimate:.5f} ({r.p_value:.4f})" if np.isfinite(r.estimate) else "계산 불가"
                rows.append(row)
            lines += [PAGE_BREAK, f"## {NAMES[method]} — {TARGETS[target]}", "", markdown_table(pd.DataFrame(rows)), "",
                      f"![{method}]({folder.relative_to(ROOT).as_posix()}/report_ACRT.png)", ""]
    kernel_audits = [json.loads((dest / "years" / str(y) / "kernel_audit.json").read_text()) for y in plan["years"]]
    a = kernel_audits[0]
    lines += [PAGE_BREAK, "## 구현·진단과 해석", "",
              f"PS2는 실제 151개 산업의 D 및 D−0.0005와 별도 ATET 비교쌍을 합친 {a['evaluated_points']}개 처치점에서 별도 RF를 학습했습니다. 최초 {a['fitted_models']}개 모형을 순차 학습·예측 후 버렸으며, 산업별 OOF 예측만 보관합니다. GPS 입력·처치·Fold·bandwidth가 같을 때 이 예측을 연도 사이에 재사용했습니다.", "",
              "직접 커널 GPS의 표적은 E[K_h(D−d)|X]로 유한 bandwidth에서 평활화된 밀도입니다. R np와 같은 √5h 반폭, 양쪽 가중치 트리밍, 밀도 하한 없음·ESS 진단만 사용을 유지했습니다. 표적 전체가 0인 학습 Fold는 RF도 0을 예측하므로 상수 0 예측으로 처리했고, 해당 처치점이 활성 지원에서 GPS 0을 만들면 계산 실패로 기록했습니다.", "",
              "GPS 변수: " + ", ".join(plan["config"]["gps_features"]) + ". 동일 산업 더미를 Mu에도 추가했습니다.", "",
              "Mu 기업 변수: " + ", ".join(plan["config"]["mu_firm_features"]) + ". Mu에는 D를 별도 추가합니다.", "",
              "PS1 GPS XGBoost: " + json.dumps(plan["config"]["gps_params"]) + ".", "",
              "PS2 GPS RF: " + json.dumps(plan["kernel_rf_params"]) + ".", "",
              "Mu XGBoost: " + json.dumps(plan["config"]["mu_params"]) + ".", "",
              "결측 대체와 표시변수는 각 학습 Fold에서만 적합했습니다. 독립 GPS/Mu 산업 Fold는 각자의 검증 산업을 학습에서 제외합니다. 이 적용의 산업 군집·서로 다른 공변량 집합·ACRT 유한차분은 저자 R 함수 그대로의 구현과 구분합니다. 산업 군집 기여도를 먼저 합산한 뒤 SE를 계산했고, 점별 SE의 단순 평균은 사용하지 않았습니다.", "",
              "표와 그래프는 보정 전 점별 p값·95% 구간입니다. 기존 결과를 보고 선택한 1538번의 후속 방법 비교이며 확증적 가설 검정이 아닙니다. PS2 회귀의 0 예측·지원 부족과 유한차분의 민감도는 결과 자료에서 함께 확인해야 합니다. 방법 변경으로 다른 평균 대상이 된 결과를 서로 직접 비교하지 않습니다.", "",
              f"계획 SHA256: `{summary['plan_sha256']}`", "", f"기존 추정 코어 SHA256: `{summary['core_source_sha256']}`", ""]
    md = exports / "Paper_1538_Observed_Comparison_ko.md"
    md.write_text("\n".join(lines))
    for ext, extra in [("pdf", ["--pdf-engine=xelatex", "-V", "mainfont=Noto Sans CJK KR", "-V", "sansfont=Noto Sans CJK KR",
                                "-V", "geometry:margin=20mm", "-V", "fontsize=10pt", "-V", "papersize=a4",
                                "-V", r"header-includes=\setlength{\emergencystretch}{3em}\sloppy"]), ("docx", [])]:
        subprocess.run(["pandoc", str(md), *extra, "-o", str(exports / f"Paper_1538_Observed_Comparison_ko.{ext}")], cwd=ROOT, check=True)
    for name in ["plan.json", "progress.json", "sample.json", "estimates.csv", "target_availability.csv", "kernel_gps_oof.npz"]:
        shutil.copy2(dest / name, exports / name)
    archive = exports / "Paper_1538_Observed_Results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in exports.rglob("*"):
            if p.is_file() and p != archive and p != md and p.suffix not in [".pdf", ".docx"]:
                z.write(p, p.relative_to(exports))
        z.writestr(md.name, md.read_text().replace(exports.relative_to(ROOT).as_posix() + "/", ""))
        for p in sorted((dest / "years").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(dest))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["checkpoint", "report"])
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    print(checkpoint(args.out) if args.command == "checkpoint" else json.dumps(render(args.out), ensure_ascii=False))
