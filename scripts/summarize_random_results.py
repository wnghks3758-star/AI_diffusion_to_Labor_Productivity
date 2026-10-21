"""Generate a Korean results note from a completed random-search batch."""
from pathlib import Path
import argparse
import json
import os
from collections import Counter
import pandas as pd


def summarize(root,output,drive_url=None):
    root=Path(root);output=Path(output);summary=json.loads((root/'summary.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    if summary['stop_reason']=='running':raise ValueError('Finish the batch before writing the final note')
    records=[json.loads(line) for line in (root/'all_attempts.jsonl').read_text().splitlines() if line]
    accepted=[r for r in records if r['status']=='accepted']
    table=pd.read_csv(root/'accepted_estimates.csv') if accepted else pd.DataFrame()
    offset=plan.get('generator_offset',0)
    text=[f"# 무작위 탐색 추가 결과: 후보 {offset+1}~{offset+plan['max_attempts']}",'',
        f"이번 배치 {summary['attempted']}/{summary['planned_max_attempts']}개를 시도했습니다. 상세 저장 {summary['accepted_detailed_cases']}개, 조건 탈락 {summary['rejected']}개, 실행 오류 {summary['failed']}개입니다.",
        f"종료 사유: `{summary['stop_reason']}`. 전체 목표 충족 {len(summary['nominal_pattern_matches'])}개, 우수 조건 포함 {len(summary['nominal_preferred_matches'])}개입니다.", '',
        '사전 2020·2021년의 전체/SME/Large/조절효과 8개가 모두 p≥.05이고, 2023년 전체 효과가 음수이며 p<.05인 경우만 상세 보관했습니다. 이후 회복·SME의 조기 양수 전환·음수 조절효과 경로까지 만족해야 전체 목표입니다. SME 2025년 유의한 양수는 우수 조건입니다.','']
    cumulative_path=root/'cumulative/cumulative_summary.json'
    if cumulative_path.exists():
        c=json.loads(cumulative_path.read_text())
        text += ['## 두 배치 누적','',
            f"누적 무작위 시도 {c['random_attempted']}개, 상세 보관 {c['retained']}개, 조건 탈락 {c['rejected']}개, 실행 오류 {c['failed']}개입니다. 누적 전체 목표 충족 {len(c['nominal_pattern_matches'])}개입니다.",
            f"두 배치 계획 후보 {c['unique_configuration_count']}개의 설정 식별자가 모두 다릅니다. 원래 난수 흐름을 이어갔으며 첫 배치의 설정·결과를 보존했습니다.",'']
    text += ['## 이번 배치의 상세 보관 사례','',
        '| 후보 | 2023 전체 추정치 (p) | 2025 SME 추정치 (p) | 2025 조절효과 추정치 (p) | 전체 목표 |',
        '|---|---:|---:|---:|---|']
    for record in accepted:
        d=table.loc[table.run_id.eq(record['run_id'])].set_index(['year','effect']);cells=[]
        for key in [(2023,'total'),(2025,'sme'),(2025,'moderation')]:
            row=d.loc[key];cells.append(f'{row.estimate:.4f} ({row.p_value:.4f})')
        text.append('| '+record['run_id']+' | '+' | '.join(cells)+' | '+str(record['pattern']['pattern_match'])+' |')
    text += ['', '표는 명목 양측 p값입니다. 판정은 반올림 전 값으로 수행했습니다.',
        f"이번 배치 {summary['random_budget_family_tests']:,}개 검정 및 이전 {plan['prior_specifications']}개 설정을 포함한 {summary['cumulative_budget_family_tests']:,}개 검정의 Bonferroni 보정을 CSV와 보고서에 함께 수록했습니다.",'',
        '## 저장 조건 통과 사례 안의 후속 조건별 충족 수','']
    for flag in ['recovery','sme_crosses_positive_earlier','sme_2025_significantly_positive','moderation_negative_significant']:
        text.append(f"- {flag}: {sum(bool(r['pattern'][flag]) for r in accepted)}개")
    text += ['', '탈락 사례의 후속 연도는 계산을 생략하므로 위 수는 전체 후보의 후속 효과 빈도가 아닙니다.','',
        '## 오류와 사용 자원','']
    for reason,count in Counter(r.get('reason','') for r in records if r['status']=='failed').items():
        text.append(f'- {reason}: {count}개')
    text += ['',f"Worker 실행 누적 {summary.get('elapsed_worker_seconds',0):.1f}초, 최대 RSS {summary.get('peak_rss_mib',0):.1f} MiB입니다. 프로세스 시작·보고서 작성·업로드 등의 전체 대기 시간과 구분합니다.",'',
        '## 상세 기록','']
    for label,path in [('실제 종료 상태',root/'summary.json'),('모든 후보 설정',root/'plan.json'),
        ('시도 장부',root/'all_attempts.csv'),('통과 사례 추정치',root/'accepted_estimates.csv'),
        ('누적 요약',cumulative_path),('누적 통과 사례 추정치',root/'cumulative/all_retained_estimates.csv')]:
        if path.exists():text.append(f'- [{label}]({os.path.relpath(path,output.parent)})')
    if drive_url:text.append(f'- [Drive 결과·보고서]({drive_url})')
    text += ['', '탐색 설계·재시작 및 보정은 RANDOM_CONTINUATION_KO.md를 참조하세요. 결과를 보고 선택한 탐색 사례이며 사전 비유의는 평행추세의 입증이 아닙니다. 조건부 부트스트랩 및 식별 제한은 METHODS_KO.md와 RANDOM_SEARCH_KO.md에 설명했습니다.']
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(text)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/random201_400')
    parser.add_argument('--output',default='docs/RANDOM_RESULTS_201_400_KO.md');parser.add_argument('--drive-url')
    args=parser.parse_args();print(json.dumps(summarize(args.root,args.output,args.drive_url),ensure_ascii=False))
