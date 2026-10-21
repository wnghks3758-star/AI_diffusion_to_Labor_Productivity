"""Write a concise final note and parent-level audit for the frozen local screen."""
from pathlib import Path
import json
import pandas as pd


def main():
    root=Path('results/adaptive200');plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    combined=json.loads(Path('results/integrated/summary.json').read_text())
    entries={e['run_id']:e for e in plan['entries']}
    records=[json.loads(line) for line in (root/'all_attempts.jsonl').read_text().splitlines() if line]
    df=pd.DataFrame([{**r,'anchor_run_id':entries[r['run_id']]['anchor_run_id']} for r in records])
    rows=[]
    for anchor in pd.read_csv(root/'selected_anchors.csv').run_id:
        subset=df.loc[df.anchor_run_id.eq(anchor)]
        rows.append({'anchor_run_id':anchor,'planned':20,'attempted':len(subset),
            **{status:int(subset.status.eq(status).sum()) for status in ['accepted','rejected','failed']},
            'nominal_pattern_matches':sum(bool(r.get('pattern',{}).get('pattern_match'))
                for r in subset.to_dict('records') if r['status']=='accepted')})
    pd.DataFrame(rows).to_csv(root/'anchor_outcomes.csv',index=False)
    lines=['# 기준 설정 주변의 추가 탐색과 전체 통과 사례 통합 결과','',
        f"이번 후보 {summary['attempted']}/200개 시도: 상세 저장 {summary['accepted_detailed_cases']}개, 조건 탈락 {summary['rejected']}개, 실행 오류 {summary['failed']}개.",
        f"종료 상태 `{summary['stop_reason']}`. 전체 목표 {len(summary['nominal_pattern_matches'])}개, 우수 조건 포함 {len(summary['nominal_preferred_matches'])}개.",'',
        '기존 현재 기준 통과 사례 37개 중 기준 설정 10개를 골라 각 20개씩 고정했다. 후보 100개는 한 항목, 100개는 두 항목을 바꿨다. 결과에 따라 후보를 다시 만들지 않았다. 상세 선정·변경 범위는 ADAPTIVE_LOCAL_SEARCH_KO.md와 plan.json에 있다.','',
        f"누적 계획 {combined['planned']}개, 실제 시도 {combined['attempted']}개 중 통합 문서에 기술한 사례는 {combined['retained_cases']}개다. 최초 설정 32개 중 2개, 변수 비교 32개 중 2개, 무작위 800개 중 33개와 이번 저장 사례를 합쳤다. 상세 표의 주요 추정치는 {combined['verified_primary_rows']:,}행이다.",'',
        '모든 상세 사례는 사전 2020·2021년 8개 주요 효과 비유의와 2023년 전체 효과의 유의한 음수를 통과했다. 기존 문서처럼 사례마다 연도별 표 1쪽과 학습 변수·설정·그림 1쪽을 제공한다. 미통과 결과는 상세 기술하지 않는다.','',
        '## 기준 설정별 시도 내역','',
        '| 기준 설정 | 시도 | 상세 보관 | 탈락 | 오류 | 전체 목표 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['anchor_run_id']} | {r['attempted']} | {r['accepted']} | {r['rejected']} | {r['failed']} | {r['nominal_pattern_matches']} |")
    accepted=[r for r in records if r['status']=='accepted']
    lines+=['','## 이번 저장 사례 안의 후속 조건','']
    for key in ['recovery','sme_crosses_positive_earlier','sme_2025_significantly_positive','moderation_negative_significant']:
        lines.append(f"- {key}: {sum(bool(r['pattern'][key]) for r in accepted)}개")
    lines+=['','후속 연도는 탈락 사례에서 계산을 생략하므로 이 수는 전체 후보의 효과 빈도가 아니다.','',
        f"이번 배치는 4,000개, 누적 계획은 {combined['family_tests']:,}개 검정을 반영한 Bonferroni 값을 함께 제공한다. 선택에 따른 탐색 결과이며, 사전 비유의와 보정이 평행추세나 인과 식별을 입증하지 않는다.",'',
        f"Worker 누적 실행 {summary.get('elapsed_worker_seconds',0):.1f}초, 최대 RSS {summary.get('peak_rss_mib',0):.1f} MiB. 전체 대기 시간과 구분한다.",'',
        '- [현재 전체 통합 현황](../results/integrated/summary.json)',
        '- [모든 후보 감사 장부](../results/integrated/all_specifications_audit.csv)',
        '- [통합 통과 추정치](../results/integrated/accepted_estimates.csv)',
        '- [새 후보 계획과 기준 설정 대비 변경](../results/adaptive200/plan.json)',
        '- [Drive 통합 문서·결과](https://drive.google.com/drive/folders/1iZU_CQbECW3MwUygY_0CW3nPrnG1GPA5)']
    Path('docs/ADAPTIVE_INTEGRATED_RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'local':summary,'integrated':combined},ensure_ascii=False))


if __name__=='__main__':main()
