"""Year plots and a small self-contained HTML report."""
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd


def write_report(table,sample,c,out):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(c.cache_dir)/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image_files = []
    for reference,part in table.groupby("reference_raw"):
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
        for ax,effects in zip(axes,[["total"],["sme","large"],["moderation"]]):
            for effect in effects:
                sub=part[part.effect.eq(effect)&part.status.eq("completed")]
                if sub.empty:
                    continue
                sub=pd.concat([sub,pd.DataFrame([dict(year=c.base_year,estimate=0.,ci_lower=0.,ci_upper=0.)])],ignore_index=True).sort_values("year")
                ax.errorbar(sub.year.to_numpy(),sub.estimate.to_numpy(),
                    yerr=np.vstack([(sub.estimate-sub.ci_lower).to_numpy(),(sub.ci_upper-sub.estimate).to_numpy()]),
                    fmt="o-",capsize=3,label=effect)
            ax.axhline(0,color="gray",ls="--",lw=.8)
            ax.axvline(c.base_year,color="gray",ls=":",lw=.8)
            ax.set(xlabel="Year",ylabel="ATET overall")
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.suptitle(f"PS{c.psmethod} | reference={100*reference:g}% | {c.sample_exclusion} | {c.support_policy}")
        name=f"ATET_overall_reference_{100*reference:g}.png"
        fig.savefig(out/name,dpi=150)
        plt.close(fig)
        image_files.append(name)
    content='<!doctype html><html lang="ko"><meta charset="utf-8"><title>ATET overall</title>'
    content+='<style>body{font-family:sans-serif;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}</style>'
    content+='<h1>고정 기준점 대비 ATET<sup>o</sup></h1>'
    content+='<p>기준점 초과 기업을 대상으로 개별 처치강도의 ATET를 기업 비중으로 평균합니다. 신뢰구간은 산업별 점수를 먼저 합산해 계산하며, 집계 가중치와 포함 지점은 고정된 것으로 취급합니다. p값·구간은 점별 값입니다. 실패 지점은 atet_by_dose.csv, strict·supported 전체 결과는 estimates.csv를 확인하십시오.</p>'
    cols=["reference_percent","year","effect","estimate","se","p_value","ci_lower","ci_upper","coverage","n_target_firms","status"]
    content+=table[cols].to_html(index=False)
    content+=''.join(f'<img src="{html.escape(name)}">' for name in image_files)
    content+=f'<p>분석 표본 기업 수: {sample["analysis_firms"]}. 제외 정책: {html.escape(c.sample_exclusion)}.</p></html>'
    (out/"report.html").write_text(content,encoding="utf-8")
