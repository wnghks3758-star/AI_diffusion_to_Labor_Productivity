"""Collect gate-passing SME-positive / negative-moderation cases across methods."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.io import file_hash, write_json
from paper_did.pipeline import ROOT
from report_paper_search import PAGE_BREAK, markdown_table, report_plot

EFFECTS = ["total", "sme", "large", "moderation"]
YEARS = [2020, 2021, 2023, 2024, 2025]
DEFAULT_ROOTS = {
    "legacy": Path("/workspace/AI_diffusion_to_Labor_Productivity/results"),
    "paper": ROOT / "audit/results",
}
OUT = ROOT / "audit/exports/Focused_Integrated_800"


def primary_table(table):
    """Select the original finite-difference estimand, excluding ATET rows."""
    if "estimand" not in table:
        return pd.DataFrame()
    estimand = next(
        (
            x
            for x in ["finite_difference_ACRT", "finite_difference_ACR"]
            if table.estimand.eq(x).any()
        ),
        None,
    )
    return table[table.estimand.eq(estimand) & table.effect.isin(EFFECTS)].copy()


def selection(table, alpha=0.05):
    t = primary_table(table)
    if (
        len(t) != 20
        or t.duplicated(["year", "effect"]).any()
        or set(t.year) != set(YEARS)
        or not np.isfinite(t[["estimate", "p_value"]].to_numpy()).all()
        or not t.p_value.between(0, 1).all()
    ):
        return None
    indexed = t.set_index(["effect", "year"])
    pre = t[t.year.isin([2020, 2021])]
    initial = indexed.loc[("total", 2023)]
    gates = (
        len(pre) == 8
        and pre.p_value.ge(alpha).all()
        and initial.estimate < 0
        and initial.p_value < alpha
    )
    sme = indexed.loc[("sme", 2025)]
    moderation = indexed.loc[("moderation", 2025)]
    return dict(
        gate_pass=bool(gates),
        selected=bool(
            gates
            and sme.estimate > 0
            and moderation.estimate < 0
            and moderation.p_value < alpha
        ),
        sme2025=float(sme.estimate),
        sme2025_p=float(sme.p_value),
        sme2025_positive_significant=bool(sme.estimate > 0 and sme.p_value < alpha),
        moderation2025=float(moderation.estimate),
        moderation2025_p=float(moderation.p_value),
        total2023=float(initial.estimate),
        total2023_p=float(initial.p_value),
        minimum_coverage=float(t.coverage.min()) if "coverage" in t else None,
    )


def collect(roots=DEFAULT_ROOTS):
    """Deduplicate archived copies by config; keep distinct settings with equal results."""
    cases = {}
    for method, root in roots.items():
        root = Path(root)
        paths = sorted(root.rglob("estimates.csv"))
        # Original paper validation exports use prefixed config/estimate filenames.
        paths += sorted((root / "paper_aligned").glob("*_estimates.csv"))
        for path in paths:
            if any(x in path.parts for x in [".scratch", "years", "cumulative"]):
                continue
            config_path = path.with_name(
                path.name.replace("estimates.csv", "config.json")
            )
            if not config_path.exists():
                continue
            table = pd.read_csv(path)
            flags = selection(table)
            if flags is None:
                continue
            config = json.loads(config_path.read_text())
            normalized = {
                k: v for k, v in config.items() if k not in ["label", "cache_dir"]
            }
            key = hashlib.sha256(
                (method + json.dumps(normalized, sort_keys=True)).encode()
            ).hexdigest()
            label = config.get("label", path.parent.name)
            source = f"{method}/{path.relative_to(root).as_posix()}"
            if key in cases:
                cases[key]["aliases"].append(dict(label=label, source=source))
                continue
            t = primary_table(table).sort_values(["year", "effect"])
            result_key = hashlib.sha256(
                t[["year", "effect", "estimate", "p_value"]]
                .to_csv(index=False, float_format="%.12g")
                .encode()
            ).hexdigest()
            cases[key] = dict(
                label=label,
                method=method,
                source=source,
                path=path,
                config_path=config_path,
                config=config,
                table=t,
                flags=flags,
                fingerprint=key,
                result_fingerprint=result_key,
                aliases=[],
                estimates_sha256=file_hash(path),
                config_sha256=file_hash(config_path),
            )
    return list(cases.values())


def render(out=OUT, roots=DEFAULT_ROOTS, batch=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = collect(roots)
    selected = sorted(
        [c for c in cases if c["flags"]["selected"]],
        key=lambda c: (c["method"], c["flags"]["moderation2025_p"], c["label"]),
    )
    summary = dict(
        scope="All recoverable complete saved configurations; archived copies deduplicated",
        filter="2020/2021 all four p>=.05; 2023 total negative p<.05; SME2025>0; Large-SME2025<0 p<.05",
        alpha=0.05,
        unique_complete_saved_configs=len(cases),
        basic_gate_passed=sum(c["flags"]["gate_pass"] for c in cases),
        focused_count=len(selected),
        focused_by_method=dict(Counter(c["method"] for c in selected)),
        unique_result_tables=len({c["result_fingerprint"] for c in selected}),
        significant_positive_sme2025=sum(
            c["flags"]["sme2025_positive_significant"] for c in selected
        ),
        source_roots={k: str(v) for k, v in roots.items()},
        new_batch=json.loads((Path(batch) / "progress.json").read_text())
        if batch
        else None,
    )
    rows, estimates = [], []
    for c in selected:
        folder = out / "cases" / c["method"] / c["label"]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c["config_path"], folder / "config.json")
        c["table"].to_csv(folder / "estimates.csv", index=False)
        report_plot(c["table"], folder, c["config"])
        rows.append(
            dict(
                label=c["label"],
                method=c["method"],
                **c["flags"],
                fingerprint=c["fingerprint"],
                result_fingerprint=c["result_fingerprint"],
                source=c["source"],
                estimates_sha256=c["estimates_sha256"],
                config_sha256=c["config_sha256"],
                aliases=json.dumps(c["aliases"], ensure_ascii=False),
            )
        )
        estimates.append(c["table"].assign(label=c["label"], method=c["method"]))
    pd.DataFrame(rows).to_csv(out / "selected_cases.csv", index=False)
    if estimates:
        pd.concat(estimates, ignore_index=True).to_csv(
            out / "selected_estimates.csv", index=False
        )
    write_json(out / "summary.json", summary)
    lines = [
        "# 저장 기준·SME 2025 양수·2025년 조절효과 유의: 통합 축약 보고서",
        "",
        f"중복 보관본을 제외한 요청 조건 충족 설정은 **{len(selected)}개**입니다. "
        f"서로 다른 결과표는 {summary['unique_result_tables']}개이며, 동일한 결과를 내는 서로 다른 설정도 설정별로 수록했습니다. "
        f"이 중 SME 2025가 유의한 양수인 설정은 **{summary['significant_positive_sme2025']}개**입니다.",
        "",
        "선정 기준은 ① 2020·2021년 전체/SME/Large/Large−SME 8개 검정 모두 p≥0.05, "
        "② 2023년 전체 추정치<0 및 p<0.05, ③ 2025년 SME 추정치>0, "
        "④ 2025년 Large−SME<0 및 p<0.05입니다. SME의 양수 자체에는 유의성을 요구하지 않았습니다. "
        "모든 p값은 별도 표시가 없으면 보정 전 값입니다. 전체 회복 가설 충족과는 다른 축약 선정 기준입니다.",
        "",
        "이전 방식(legacy)과 논문 방식(paper)을 분리합니다. 이전 방식은 기존 이분산 GPS·밀도 하한·당해연도 통제 설정 등이 포함된 역사적 결과입니다. "
        "논문 방식은 2022년 기준 통제, 산업 Fold·군집 추론, psmethod=1 공통 Fold 분산 GPS, 밀도 하한 없음·ESS 진단만 사용합니다. "
        "두 방식의 수치를 같은 추정법의 반복 결과로 해석하거나 합산하지 않습니다.",
        "",
        "자료 범위는 현 작업공간에 원래 설정과 20개 연도별 주효과가 함께 보존되어 재검증 가능한 결과입니다. "
        "삭제된 탈락 추정치는 복원하거나 나열하지 않습니다. 원래 설정 JSON을 그대로 제공하며 추정치를 새로 계산하지 않았습니다.",
        "",
        "## 추가 200개 탐색",
        "",
    ]
    if summary["new_batch"]:
        b = summary["new_batch"]
        new_selected = sum(
            c["method"] == "paper" and 601 <= int(c["label"].rsplit("_", 1)[-1]) <= 800
            for c in selected
            if c["label"].startswith("paper_search_")
        )
        lines += [
            f"601~800번: {b['attempted']}/200개 실행, 상태 {b['status']}. "
            f"기본 저장 기준 통과 {len(b['retained'])}개, 본 보고서의 추가 조건 충족 {new_selected}개.",
            "",
            "저장 기준에 실패하면 해당 설정을 즉시 중단하고 추정치·로그·학습 캐시를 삭제했습니다. "
            "확정 설정표와 중단 사유별 건수만 남깁니다. "
            "새 탐색은 이전 600개와 중복되지 않으며, 변수별 포함 여부·학습기·Fold·산업 더미·bandwidth·trim을 변경했습니다.",
            "",
            markdown_table(
                pd.DataFrame(list(b["counts"].items()), columns=["집계", "건수"])
            ),
            "",
        ]
    lines += ["## 조건 충족 설정 목록", ""]
    for method in ["legacy", "paper"]:
        group = [c for c in selected if c["method"] == method]
        lines += [
            f"### {'이전 방식' if method == 'legacy' else '논문 방식'}: {len(group)}개",
            "",
        ]
        if not group:
            lines += ["요청한 추가 조건까지 충족한 설정이 없습니다.", ""]
        else:
            overview = [
                [
                    c["label"],
                    f"{c['flags']['sme2025']:.4f} ({c['flags']['sme2025_p']:.4f})",
                    f"{c['flags']['moderation2025']:.4f} ({c['flags']['moderation2025_p']:.4f})",
                ]
                for c in group
            ]
            lines += [
                markdown_table(
                    pd.DataFrame(
                        overview, columns=["설정", "SME2025 (p)", "Large−SME2025 (p)"]
                    )
                ),
                "",
            ]
    for c in selected:
        cfg, t, f = c["config"], c["table"], c["flags"]
        folder = f"cases/{c['method']}/{c['label']}"
        mu_ind = cfg.get(
            "mu_industry_features",
            [
                v
                for v in cfg.get("mu_features", [])
                if v.startswith("ind_")
                or v.startswith("ln_ind")
                or v == "ln_cum_patent"
            ],
        )
        mu_firm = cfg.get(
            "mu_firm_features",
            [v for v in cfg.get("mu_features", []) if v not in mu_ind],
        )
        lines += [
            PAGE_BREAK,
            f"## {c['label']} ({c['method']})",
            "",
            f"GPS {cfg.get('gps_model')} / Mu {cfg.get('mu_model')}; BW 배율 {cfg.get('bandwidth_factor', cfg.get('bandwidth_scale'))}; "
            f"trim {cfg.get('trim', cfg.get('trim_threshold'))}; Fold {cfg.get('n_splits')}; seed {cfg.get('seed')}; "
            f"통제 시점 {cfg.get('covariate_time')}. 최소 coverage {f['minimum_coverage']:.2%}.",
            "",
            f"산업 분류 {cfg.get('industry_scheme')}, 최소 산업수 {cfg.get('industry_min_groups')}, "
            f"최소 관측수 {cfg.get('industry_min_obs')}, 산업 더미 {cfg.get('include_industry_dummies')}. "
            f"GPS knot {cfg.get('spline_n_knots', cfg.get('gps_n_knots'))}, 차수 {cfg.get('spline_degree', cfg.get('gps_degree'))}; "
            f"GPS 파라미터 {json.dumps(cfg.get('gps_params', {}))}; Mu 파라미터 {json.dumps(cfg.get('mu_params', {}))}.",
            "",
            f"반복 Fold {cfg.get('n_repeats', 1)}; bandwidth 통계량 {cfg.get('bandwidth_statistic', 'paper_sd' if c['method'] == 'paper' else 'legacy_default')}; "
            f"밀도 하한 {cfg.get('density_floor', '없음')}; ESS {cfg.get('ess', '진단만')}; "
            f"공통하한 ESS {cfg.get('common_lower_ess', cfg.get('min_ess_prime', '해당 없음'))}; "
            f"유한차분 간격 {cfg.get('diff_step')}; 누락 구간 처리 {cfg.get('gap_policy')}. "
            "학습기 검색 공간 등 나머지 설정은 첨부 JSON 원문에 있습니다.",
            "",
            "GPS 변수: " + ", ".join(cfg.get("gps_features", [])) + ".",
            "",
            "Mu 산업 변수: " + ", ".join(mu_ind) + ".",
            "",
            "Mu 기업 변수: "
            + (", ".join(mu_firm) if cfg.get("include_firm_controls", True) else "제외")
            + ".",
            "",
        ]
        display = []
        for year in YEARS:
            row = [year]
            for effect in EFFECTS:
                v = t[(t.year == year) & (t.effect == effect)].iloc[0]
                row.append(f"{v.estimate:.4f} ({v.p_value:.4f})")
            display.append(row)
        lines += [
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
            f"![{c['label']}]({folder}/report_ACRT.png)",
            "",
            f"[전체 설정 JSON]({folder}/config.json) · [추정 결과 CSV]({folder}/estimates.csv)",
            "",
        ]
    lines += [
        PAGE_BREAK,
        "## 해석과 재현 파일",
        "",
        "조건에 맞춘 반복 탐색과 사후 선정 결과입니다. 선택된 원래 p값을 확증적 유의성으로 읽을 수 없습니다. "
        "사전 비유의 역시 평행추세의 증명이 아닙니다. 위 조건에는 단조 회복·더 빠른 양수 전환 등 전체 가설의 나머지 요건을 추가하지 않았습니다.",
        "",
        "논문 방식은 계획된 800개×20개=16,000개 검정 기준 누적 Bonferroni 보정값을 새 배치 전체 결과에 제공합니다. "
        "과거 방식의 전체 탐색 규모는 그 보정에 포함되지 않습니다. 이전 설정의 기존 보정 열은 원본 CSV에 유지하며 서로 다른 검정 집합의 보정값을 혼합하지 않습니다. "
        "표본·모형 가정·ACRT 변환과 산업 군집 확장에 관한 기존의 한계도 유지됩니다.",
        "",
        "selected_cases.csv: 조건 충족 설정·p값·원본 해시·보관본 별칭. selected_estimates.csv: 해당 설정의 20개 주효과. "
        "cases/: 설정 JSON·추정 결과·그래프. summary.json: 대상 범위·건수. 원자료·기업별 관측치·학습 모델은 포함하지 않습니다.",
        "",
    ]
    md = out / "Focused_Integrated_800_ko.md"
    md.write_text("\n".join(lines))
    for ext, extra in [
        (
            "pdf",
            [
                "--pdf-engine=xelatex",
                "-V",
                "mainfont=Noto Sans CJK KR",
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
            ["pandoc", md.name, *extra, "-o", f"Focused_Integrated_800_ko.{ext}"],
            cwd=out,
            check=True,
        )
    archive = out / "Focused_Integrated_800_Results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p != archive:
                z.write(p, p.relative_to(out))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.out, batch=args.batch), ensure_ascii=False, indent=2))
