"""Aggregate retained search cases, export Korean report and safe checkpoints."""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from labor_did.io import file_hash, write_json
from search_paper import CORE, DEST, PRIMARY, ROOT, evaluate, primary

PAGE_BREAK = '\n\n```{=latex}\n\\clearpage\n```\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'


def markdown_table(frame):
    # Avoid an additional optional tabulate dependency.
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([head, rule, *rows])


def report_plot(table, folder, config):
    """Stack three panels so their labels remain readable in an A4 document."""
    from labor_did.plot import plt

    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.2, 5.6), sharex=True, layout="constrained"
        )
        for ax, effects, label in zip(
            axes,
            [["total"], ["sme", "large"], ["moderation"]],
            ["Total ACRT", "SME / Large ACRT", "Large - SME"],
        ):
            for effect in effects:
                part = table[table.effect.eq(effect)]
                part = pd.concat(
                    [
                        part,
                        pd.DataFrame(
                            [dict(year=2022, estimate=0.0, ci_lower=0.0, ci_upper=0.0)]
                        ),
                    ]
                ).sort_values("year")
                ax.errorbar(
                    part.year,
                    part.estimate,
                    yerr=[part.estimate - part.ci_lower, part.ci_upper - part.estimate],
                    fmt="o-",
                    capsize=3,
                    label=effect,
                )
            ax.axhline(0.0, color="gray", ls="--", lw=0.8)
            ax.axvline(2022, color="gray", ls=":", lw=0.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if len(effects) > 1:
                ax.legend(ncol=2, loc="best")
        axes[-1].set(xlabel="Year", xticks=[2020, 2021, 2022, 2023, 2024, 2025])
        fig.suptitle(f"{config['label']} | exploratory ACRT, 95% pointwise CI")
        fig.savefig(folder / "report_ACRT.png", dpi=180)
        plt.close(fig)


def checkpoint(dest=DEST):
    dest = Path(dest).resolve()
    progress = json.loads((dest / "progress.json").read_text())
    path = dest.parent / f"{dest.name}_checkpoint_{progress['attempted']:03d}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "progress.json", json.dumps(progress, ensure_ascii=False, indent=2)
        )
        for name in [
            "plan.json",
            "plan.original.json",
            "runner.original.py",
        ]:
            if (dest / name).exists():
                archive.write(dest / name, name)
        for label in progress["retained"]:
            for path_in in sorted((dest / "retained" / label).rglob("*")):
                if path_in.is_file():
                    archive.write(path_in, path_in.relative_to(dest))
    return path


def report(dest=DEST, exports=None):
    dest = Path(dest).resolve()
    exports = Path(exports or ROOT / "audit/exports" / dest.name)
    exports.mkdir(parents=True, exist_ok=True)
    plan = json.loads((dest / "plan.json").read_text())
    stem = plan.get("report_stem", "Paper_Search_200")
    image_root = dest.relative_to(ROOT / "audit").as_posix()
    progress = json.loads((dest / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["attempted"] == 200
    rows, tables = [], []
    for label in progress["retained"]:
        folder = dest / "retained" / label
        table = pd.read_csv(folder / "estimates.csv")
        raw = evaluate(table)
        adjusted = evaluate(table, "p_bonferroni_search")
        initial = primary(table).query("year == 2023 and effect == 'total'").iloc[0]
        config = json.loads((folder / "config.json").read_text())
        rows.append(
            dict(
                label=label,
                **raw,
                adjusted_pattern_match=adjusted["pattern_match"],
                adjusted_preferred_match=adjusted["preferred_match"],
                cumulative_pattern_match=(
                    evaluate(table, "p_bonferroni_cumulative")["pattern_match"]
                    if "p_bonferroni_cumulative" in table
                    else None
                ),
                minimum_coverage=float(primary(table).coverage.min()),
                gps_model=config["gps_model"],
                mu_model=config["mu_model"],
                bandwidth_factor=config["bandwidth_factor"],
                trim=config["trim"],
                total_2023_estimate=float(initial.estimate),
                total_2023_p=float(initial.p_value),
            )
        )
        tables.append(table.assign(label=label))
    columns = [
        "label",
        "complete",
        *CORE,
        "sme_crosses_positive_earlier",
        "sme_2025_significantly_positive",
        "pattern_match",
        "preferred_match",
        "criteria_passed",
        "sme_2025_estimate",
        "sme_2025_p",
        "adjusted_pattern_match",
        "adjusted_preferred_match",
        "cumulative_pattern_match",
        "minimum_coverage",
        "gps_model",
        "mu_model",
        "bandwidth_factor",
        "trim",
        "total_2023_estimate",
        "total_2023_p",
    ]
    columns = list(dict.fromkeys(columns))
    flags = pd.DataFrame(rows).reindex(columns=columns)
    flags.to_csv(dest / "retained_flags.csv", index=False)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            dest / "retained_estimates.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "label",
                "year",
                "estimand",
                "effect",
                "estimate",
                "p_value",
                "p_bonferroni_search",
            ]
        ).to_csv(dest / "retained_estimates.csv", index=False)
    specs = []
    for entry in plan["entries"]:
        c = entry["config"]
        specs.append(
            {
                "number": entry["number"],
                "family": entry["family"],
                "family_name": entry["family_name"],
                **{
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in c.items()
                },
            }
        )
    pd.DataFrame(specs).to_csv(dest / "planned_settings.csv", index=False)
    matches = int(flags.pattern_match.sum()) if len(flags) else 0
    preferred = int(flags.preferred_match.sum()) if len(flags) else 0
    summary = dict(
        attempted=200,
        retained=len(flags),
        pattern_matches=matches,
        preferred_matches=preferred,
        adjusted_pattern_matches=int(flags.adjusted_pattern_match.sum())
        if len(flags)
        else 0,
        adjusted_preferred_matches=int(flags.adjusted_preferred_match.sum())
        if len(flags)
        else 0,
        counts=progress["counts"],
        plan_sha256=file_hash(dest / "plan.json"),
        family_size=4000,
        overall_source_sha256=plan["source_sha256"],
        data_sha256=plan["data_sha256"],
        scope="This batch only; earlier searches excluded from multiplicity correction.",
    )
    if "cumulative_count" in plan:
        summary.update(
            cumulative_attempted=plan["cumulative_count"],
            cumulative_retained=plan["previous_summary"].get(
                "cumulative_retained", plan["previous_summary"]["retained"]
            )
            + len(flags),
            cumulative_raw_pattern_matches=plan["previous_summary"].get(
                "cumulative_raw_pattern_matches",
                plan["previous_summary"]["pattern_matches"],
            )
            + matches,
            cumulative_family_size=plan["cumulative_correction_family_size"],
            cumulative_adjusted_pattern_matches=int(
                flags.cumulative_pattern_match.sum()
            )
            if len(flags)
            else 0,
            scope=f"This batch; cumulative Bonferroni column covers the first {plan['cumulative_count']} paper-aligned specifications only.",
        )
    write_json(dest / "summary.json", summary)
    counts = progress["counts"]
    update = plan.get("execution_order_update")
    if update:
        n = update["completed_at_update"]
        order_note = f"첫 {n}개는 2020→2021→2023 순서로 검사했습니다. 초기 하락 기준에서 중단되는 경우가 대부분이어서 남은 {200 - n}개는 2023→2020→2021 순서로 검사해 계산을 줄였습니다. 변수·모형·수치 설정 200개와 판정 기준은 바꾸지 않았으며, plan.original.json과 plan.json의 entries는 완전히 같습니다. 진행 중이던 미완료 묶음은 같은 설정으로 다시 계산했습니다. 검사를 먼저 통과하지 못하면 이후 조건은 계산하지 않으므로 중단 유형별 건수는 그 순서에 의존합니다."
    else:
        order_note = "2023년 초기 하락, 2020년 사전 조건, 2021년 사전 조건 순서로 검사했습니다. 한 조건이라도 실패하면 이후 연도는 계산하지 않습니다."
    result_line = f"200개 설정 중 사전 기준과 초기 하락 기준을 모두 통과한 설정은 {len(flags)}개입니다. 전체 가설 충족은 {matches}개, 2025년 SME 유의한 양수까지 충족한 설정은 {preferred}개입니다."
    if "cumulative_count" in plan:
        result_line += f" 이전 배치와 합하면 누적 {plan['cumulative_count']}개 중 기본 기준 통과 {summary['cumulative_retained']}개, 전체 가설 충족 {summary['cumulative_raw_pattern_matches']}개입니다."
    text = [
        "# " + plan.get("report_title_ko", "논문 방식 기반 200개 설정 탐색"),
        "",
        result_line,
        "",
        "## 탐색과 중단 기준",
        "",
        "가설 판정 대상은 기존 방식으로 연결한 finite-difference ACRT입니다. 고정 두 처치점의 ATET는 별도 결과로 보존하며 두 추정 대상을 혼용하지 않았습니다. "
        + plan.get(
            "design_description_ko",
            "200개 설정은 실행 전에 확정했습니다. 기본 설정을 제외한 199개는 비교 설정입니다. 40개 변수·학습기 조합에 각각 5개 bandwidth·트리밍 조합을 적용했습니다.",
        ),
        "",
        "2020년과 2021년의 전체·SME·Large·Large−SME가 모두 p≥0.05여야 하며, 2023년 전체 효과가 음수이고 p<0.05여야 합니다. 한 조건이라도 실패하면 즉시 중단하고 그 설정의 추정치·로그·학습 캐시는 삭제했습니다. 아래에는 통과한 설정만 기술하며, 탈락 설정별 결과는 남기지 않았습니다. 실행 전에 정한 설정 목록과 중단 유형별 건수는 재현성과 탐색 규모 확인을 위해 보존했습니다.",
        "",
        order_note,
        "",
        markdown_table(
            pd.DataFrame(
                [
                    ["총 시도", 200],
                    ["2020년 사전 기준 중단", counts.get("pre_2020", 0)],
                    ["2021년 사전 기준 중단", counts.get("pre_2021", 0)],
                    ["2023년 전체 유의한 하락 미충족", counts.get("initial_2023", 0)],
                    ["수치 계산 실패", counts.get("numerical_failure", 0)],
                    ["실행 오류", counts.get("execution_error", 0)],
                    ["기준 통과·결과 저장", len(flags)],
                ],
                columns=["구분", "건수"],
            )
        ),
        "",
        "## 변경한 변수와 설정",
        "",
        plan.get(
            "variables_description_ko",
            "GPS 산업 변수 12개, Mu 산업 변수 13개, Mu 기업 변수 5개를 각각 하나씩 제외하는 30개 조합을 기본 조합과 비교했습니다. 추가 9개 조합에서는 산업·기업 변수를 개별 확률 추출(seed 2026102303)로 포함하거나 제외했습니다. GPS의 근로시간 변수는 추가 조합에서 포함될 수도 있습니다. Mu 기업 변수는 large_dummy, small_dummy, firm_rnd_ratio, ln_firm_avg_pay, ln_firm_ast입니다. 처치 D는 Mu에 별도로 유지했습니다.",
        ),
        "",
        plan.get(
            "models_description_ko",
            "기본 GPS Spline·Mu XGBoost 외에 GPS Random Forest·Lasso·Ridge·선형·XGBoost와 Mu Random Forest·Lasso·Ridge·Spline을 포함했습니다. 추가 조합에서는 산업 Fold 수 3/4/5, seed 1/42/2026, spline knot 3/4/5·차수 2/3, 산업 분류의 최소 관측수 2/4/8·최소 산업수 2/4/6, 대분류 및 산업 더미 제외, 기업 변수 전체 제외도 비교했습니다. 개별 설정의 정확한 값과 변수 목록은 planned_settings.csv와 plan.json에 있습니다.",
        ),
        "",
        plan.get(
            "numeric_description_ko",
            "모든 변수·학습기 조합에 적용한 (bandwidth 배율, 트리밍)은 (0.70, 0.10), (0.45, 0.10), (0.55, 0.05), (0.85, 0.10), (1.00, 0.15)입니다. Bandwidth는 산업 수와 산업 처치값의 표본 표준편차로 저자 방식의 공식을 계산한 뒤 해당 배율을 곱했습니다. 배율 0.70이 저자 패키지 기준입니다.",
        ),
        "",
        "## 유지한 추정 원칙과 전체 가설",
        "",
        "2022년 기준 통제변수, 산업별 GPS·산업 Fold·산업 군집 추론, psmethod=1 정규밀도와 검증 Fold 산업 잔차의 공통 분산, 훈련 Fold 중앙값 대체·결측 표시변수, ESS 제외 기준 없음, 밀도 하한 없음, 내부 누락 구간 보간을 유지했습니다. 표본은 기존 균형 패널이며 결과의 실제 ACRT 평균 대상 비율은 각 저장 설정의 coverage에 기록합니다. 유한차분 간격 0.0005, 보조 군집 bootstrap 5,000회, Large−SME는 각 규모의 자체 기업 평균 차이입니다.",
        "",
        "전체 가설은 위 두 저장 기준에 더해 전체·SME·Large의 2023≤2024≤2025 회복(2025>2023), SME의 더 이른 양수 전환, 사후 모든 연도의 Large−SME 음수와 지속적 유의 또는 2025년 유의·2023년 대비 p값 감소를 요구합니다. 여기에 SME 2025년 유의한 양수를 추가하면 선호 조건 충족입니다. 사전 비유의는 평행추세가 성립한다는 증명이 아닙니다.",
        "",
        "## 기준을 통과한 설정",
        "",
    ]
    if not len(flags):
        text += [
            "이번 200개에서는 사전 비유의와 2023년 전체의 유의한 하락을 동시에 만족한 설정이 없었습니다. 따라서 저장 기준을 통과한 추정 결과나 전체 가설 충족 사례가 없습니다. 요청한 중단 규칙에 따라 2024·2025년 추가 추정은 수행하지 않았으며, 탈락 추정치와 개별 그래프는 저장하지 않았습니다.",
            "",
        ]
    else:
        labels = {
            "random_forest": "RF",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "xgboost": "XGB",
            "spline": "Spline",
            "linear": "Linear",
        }
        overview = []
        for row in sorted(rows, key=lambda r: (-r["criteria_passed"], r["label"])):
            overview.append(
                [
                    row["label"].rsplit("_", 1)[-1],
                    labels[row["mu_model"]],
                    f"{row['bandwidth_factor']}/{row['trim']}",
                    f"{row['total_2023_estimate']:.4f} ({row['total_2023_p']:.4f})",
                    f"{row['sme_2025_estimate']:.4f} ({row['sme_2025_p']:.4f})",
                    row["criteria_passed"],
                ]
            )
        text += [
            "아래 표는 저장 기준을 통과한 설정만 요약합니다. 괄호는 보정 전 p값이고, 충족 수는 6개 가설 조건 중 통과한 수입니다.",
            "",
            markdown_table(
                pd.DataFrame(
                    overview,
                    columns=[
                        "번호",
                        "Mu",
                        "BW/Trim",
                        "전체2023 (p)",
                        "SME2025 (p)",
                        "충족수",
                    ],
                )
            ),
            "",
        ]
        for row in sorted(rows, key=lambda r: (-r["criteria_passed"], r["label"])):
            label = row["label"]
            folder = dest / "retained" / label
            c = json.loads((folder / "config.json").read_text())
            table = primary(pd.read_csv(folder / "estimates.csv"))
            report_plot(table, folder, c)
            text += [
                PAGE_BREAK,
                f"### {label}",
                "",
                f"전체 가설: {row['pattern_match']}, 선호 조건: {row['preferred_match']}, 조건 충족 수: {row['criteria_passed']}/6, 최소 평균 대상 비율: {row['minimum_coverage']:.2%}.",
                "",
                f"GPS {c['gps_model']} / Mu {c['mu_model']}; bandwidth 배율 {c['bandwidth_factor']}; trim {c['trim']}; Fold {c['n_splits']}; seed {c['seed']}.",
                "",
                f"Spline knot {c['spline_n_knots']}·차수 {c['spline_degree']}; 산업 분류 {c['industry_scheme']}·최소 산업수 {c['industry_min_groups']}·최소 관측수 {c['industry_min_obs']}; 산업 더미 {c['include_industry_dummies']}; GPS 파라미터 {json.dumps(c['gps_params'])}; Mu 파라미터 {json.dumps(c['mu_params'])}.",
                "",
                "GPS 변수: " + ", ".join(c["gps_features"]) + ".",
                "",
                "Mu 산업 변수: " + ", ".join(c["mu_industry_features"]) + ".",
                "",
                "Mu 기업 변수: "
                + (
                    ", ".join(c["mu_firm_features"])
                    if c["include_firm_controls"]
                    else "제외"
                )
                + ".",
                "",
            ]
            display = []
            for year in [2020, 2021, 2023, 2024, 2025]:
                values = [year]
                for effect in PRIMARY:
                    r = table[table.year.eq(year) & table.effect.eq(effect)].iloc[0]
                    values.append(f"{r.estimate:.4f} ({r.p_value:.4f})")
                display.append(values)
            text += [
                markdown_table(
                    pd.DataFrame(
                        display,
                        columns=[
                            "연도",
                            "전체 (p)",
                            "SME (p)",
                            "Large (p)",
                            "Large−SME (p)",
                        ],
                    )
                ),
                "",
                f"![{label} ACRT]({image_root}/retained/{label}/report_ACRT.png)",
                "",
            ]
    text += [
        PAGE_BREAK,
        "## 탐색적 해석과 파일",
        "",
        "변수와 설정을 결과 기준으로 선택한 탐색이므로 선택된 원래 p값을 확증적 검정으로 해석할 수 없습니다. 저장 기준 자체도 결과를 선택합니다. 이번 200개×20개 ACRT 검정을 계획된 4,000개 검정으로 잡은 Bonferroni 보정값(p_bonferroni_search)을 별도로 제공합니다. 중단·삭제된 검정도 계획된 검정 수에서 제외하지 않았습니다. 이는 이번 배치에 대한 보정이며 이전의 여러 탐색까지 포함하지 않습니다. 원래 점별 추론의 정규성·식별 가정과 ACRT 변환·산업 군집 확장에 관한 한계도 남습니다.",
        "",
        plan.get("correction_note_ko", ""),
        "",
        "plan.json / planned_settings.csv: 실행 전 확정한 전체 설정. progress.json / summary.json: 실행·중단 건수. retained_flags.csv / retained_estimates.csv: 저장 기준을 통과한 설정만의 판정과 결과. retained/ 아래에는 해당 설정값, 연도별 계산 진단, ATET·ACRT 결과와 그래프가 있습니다. 학습 모델·원자료는 보고서 업로드에 포함하지 않습니다.",
        "",
        f"계획 SHA256: `{summary['plan_sha256']}`",
        "",
        f"추정 소스 SHA256: `{plan['source_sha256']}`",
        "",
    ]
    md = ROOT / "audit" / f"{stem}_ko.md"
    md.write_text("\n".join(text))
    for extension, extra in [
        (
            "pdf",
            [
                "--pdf-engine=xelatex",
                "-V",
                "mainfont=Noto Sans CJK KR",
                "-V",
                "sansfont=Noto Sans CJK KR",
                "-V",
                "geometry:margin=20mm",
                "-V",
                "fontsize=10pt",
                "-V",
                "papersize=a4",
                "-V",
                r"header-includes=\setlength{\emergencystretch}{3em}\sloppy",
            ],
        ),
        ("docx", []),
    ]:
        subprocess.run(
            [
                "pandoc",
                md.name,
                *extra,
                "-o",
                str(exports / f"{stem}_ko.{extension}"),
            ],
            cwd=md.parent,
            check=True,
        )
    for name in [
        "summary.json",
        "retained_flags.csv",
        "retained_estimates.csv",
        "planned_settings.csv",
        "plan.json",
        "progress.json",
    ]:
        shutil.copy2(dest / name, exports / name)
    archive = exports / f"{stem}_Results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file() and ".scratch" not in p.parts:
                z.write(p, p.relative_to(dest))
        z.write(md, md.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["checkpoint", "report"])
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    print(
        checkpoint(args.out)
        if args.command == "checkpoint"
        else json.dumps(report(args.out), ensure_ascii=False)
    )
