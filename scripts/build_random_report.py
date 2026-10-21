"""Build Word/PDF reports from retained random-search cases, without input data.

Run with a Python that has pandas, numpy and python-docx. PDF needs xelatex.
"""
from pathlib import Path
import argparse
import json
import os
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
import report_renderers as renderers

ROOT=Path(__file__).resolve().parents[1]
EFFECTS=['total','sme','large','moderation']
NAMES={'total':'전체','sme':'SME','large':'Large','moderation':'조절효과'}

def p(text):return {'kind':'p','text':text}
def h(text):return {'kind':'heading','text':text}
def t(headers,rows,widths=None,size=9):
    return {'kind':'table','headers':headers,'rows':rows,'widths':widths or [1]*len(headers),'size':size}
def n(value):return f'{float(value):.4f}' if pd.notna(value) else '—'
def probability(value):return '<.0001' if pd.notna(value) and value<.0001 else n(value)

def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    work=output.parent/'tex';work.mkdir(parents=True,exist_ok=True)
    for var,name in [('XDG_CACHE_HOME','cache'),('TEXMFVAR','texmf-var'),('TEXMFCONFIG','texmf-config')]:
        directory=work/name;directory.mkdir(exist_ok=True);os.environ[var]=str(directory)
    renderers.WORK=work;renderers.OUT=output
    plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'summary.json').read_text())
    ledger=[json.loads(line) for line in (root/'all_attempts.jsonl').read_text().splitlines() if line]
    accepted=[r for r in ledger if r['status']=='accepted']
    data=pd.read_csv(root/'accepted_estimates.csv') if accepted else pd.DataFrame()
    # User-facing prose is Korean; status codes are reproduced literally for audit.
    end_ko={'target_found':'전체 목표 충족 사례를 찾아 자동 종료','budget_exhausted':'최대 시도 수에 도달해 종료','running':'중간 실행 상태'}
    pages=[[
        {'kind':'title','text':'노동생산성 연속 DiD\n무작위 변수·설정 탐색 보고서'},
        p('2026년 10월 21일 · 탐색 결과 · 조건 통과 사례만 상세 보관'),
        t(['최대 / 실제 시도','상세 보관','탈락 / 실행 오류'],[[f"{plan['max_attempts']} / {summary['attempted']}",
            str(summary['accepted_detailed_cases']),f"{summary['rejected']} / {summary['failed']}" ]],size=10),
        h('실제 종료 결과'),p(end_ko[summary['stop_reason']]+f" ({summary['stop_reason']})."),
        p(f"필수 전체 목표 충족: {len(summary['nominal_pattern_matches'])}개. 2025년 SME 유의한 양수까지 충족하는 우수 사례: {len(summary['nominal_preferred_matches'])}개."),
        p('전체 목표 통과 ID: '+(', '.join(summary['nominal_pattern_matches']) or '없음')+'.'),
        h('사용자가 요청한 저장 필터'),
        p('2020·2021년의 전체·SME·Large·조절효과 8개가 모두 명목 p≥.05이고, 2023년 전체 효과가 음수이며 p<.05일 때만 상세 결과를 남겼다. 초기 SME와 Large의 하락은 요구하지 않았다.'),
        p('사전 연도에서 탈락하면 후속 연도 학습을 생략했다. 탈락한 추정치·그림은 임시 폴더에서 삭제했다. 설정·시도 번호·탈락 사유는 선택 과정을 확인할 수 있도록 장부에 남겼다.'),
        h('탐색이라는 해석 범위'),
        p('요구한 유의성 패턴을 찾은 경우에도 확인적 증거로 부르지 않는다. 기존 64개 설정을 본 뒤 추가로 실시한 탐색이며, 비유의한 사전 효과는 평행추세의 입증이 아니다.'),
        p(f"통과 사례의 주요 p값에는 전체 계획 {summary['random_budget_family_tests']:,}개 검정 및 이전 64개 설정을 포함한 {summary['cumulative_budget_family_tests']:,}개 검정의 Bonferroni 보정을 함께 제공한다.")
    ]]
    pages.append([
        h('1. 후보 생성과 고정 범위'),
        p(f"새 결과 계산 전 후보 {plan['max_attempts']}개와 기준을 고정했다. 생성기 seed={plan['generator_seed']}; 모형 seed=42. 새 결과를 보고 후보를 교체하지 않았다. 실행 오류도 시도 수에 포함한다."),
        t(['변경 항목','무작위 범위'],[
            ['GPS 변수','산업 통제 13개 중 2~13개 부분집합'],
            ['Mu 변수','산업 13개·기업 3개·규모 더미 2개 중 2~18개 부분집합'],
            ['GPS','선형 / spline; knot 3·4·5, 차수 1·2·3'],
            ['Mu','XGBoost(65%) / Ridge(20%) / spline Ridge(15%)'],
            ['트리 설정','트리 80·160·240, 깊이 2·3·4, 학습률 .03·.05·.1, 행·열 추출률 .65·.8·1 및 L1/L2 등'],
            ['Ridge','alpha .1·1·10·100; spline knot 3·4, 차수 1·2·3'],
            ['Fold','산업 분리 2·3·4·5; 반복 2·3·5; 내부 2·3'],
            ['산업 더미','포함 여부, refined·대분류·없음; 정제 산업수·관측수 기준'],
            ['ESS / 하한 ESS','3·5·8·10 / 3·5·8'],
            ['트리밍 / 밀도 하한','.05·.1·.2 / .0025·.005·.01'],
            ['대역폭 / 차분 간격','.35·.5·.7 / .002·.004'],
            ['기타','strict / bridge; 해당 연도 / 기준 연도 통제값']], [1.2,4.4],9),
        p('고정: 결과변수 ln_firm_prod, 처치 log(100×office_ratio), 기준 2022년, 산업 내 사전 매출 중앙값의 기업규모 기준, 산업 클러스터, 유의수준 .05, bootstrap 1,999회. 추가 Mu 튜닝은 하지 않는다.'),
        p('부분집합 크기와 해당 크기의 조합을 각각 균등 추출했다. 동일 시점 설계 안에서는 전체 16개 기준 통제변수의 결측 필터를 유지했다. 시점·지지 설계가 바뀌면 실현 표본이 바뀔 수 있으며 통과 사례별 해시를 남겼다.')
    ])
    rows=[]
    for r in accepted:
        d=data.loc[data.run_id.eq(r['run_id'])&data.year.eq(2025)].set_index('effect')
        rows.append([r['run_id'],f"{n(d.loc['sme','estimate'])} ({probability(d.loc['sme','p_value'])})",
            f"{n(d.loc['moderation','estimate'])} ({probability(d.loc['moderation','p_value'])})",
            '충족' if r['pattern']['pattern_match'] else '미충족'])
    pages.append([
        h('2. 저장 조건을 통과한 사례'),
        p(f"총 {len(accepted)}개 사례를 상세 보관했다. 아래 p값은 명목 양측 p값이며, 전체 목표 충족 여부와 별도로 표시했다."),
        t(['후보 ID','2025 SME 추정치 (p)','2025 조절효과 (p)','전체 목표'],rows,[1.1,1.8,1.8,.9],8.7) if rows else p('상세 저장 조건을 통과한 사례가 없으므로 상세 추정치·그림을 남기지 않았다.'),
        h('전체 목표의 정확한 판정'),
        p('저장 조건에 더해 전체·SME·Large가 각각 2023≤2024≤2025 및 2025>2023을 만족해야 한다. SME의 첫 양수 연도가 Large보다 빨라야 하며, SME만 관측 기간 안에 양수가 되어도 통과한다.'),
        p('조절효과는 사후 세 연도 모두 음수여야 한다. 세 연도 모두 유의하거나, 2025년에는 유의하고 p2025≤p2023이면 통과한다. 조절효과는 Large의 평균 ACR−SME의 평균 ACR이며 법적 SME 분류나 별도 상호작용 계수가 아니다.'),
        p('2025년 SME 유의한 양수는 우수 조건이다. 필수 전체 목표를 만족하면 이 우수 조건을 아직 만족하지 않아도 탐색을 멈춘다.'),
        h('탈락 및 실행 오류'),
        t(['분류','시도 수'],[[reason,str(count)] for reason,count in summary['rejection_counts'].items()], [4,1],9),
        p('실행 오류 사유: '+('; '.join(f'{reason} ({count}건)' for reason,count in Counter(r.get('reason','') for r in ledger if r['status']=='failed').items()) or '없음')+'.'),
        p('산업층에 산업이 1개뿐인 경우 산업 단위 층화 교차적합이 성립하지 않아 실행 오류로 남겼다. 이를 결과가 유리한 임의의 fold로 바꾸거나 다른 후보로 대체하지 않았다.')
    ])
    pages.append([
        h('3. 검증·재현·한계'),
        p('검증: 사전 p값 임계치 및 비유한 값 처리, 2023년 전체만 하락하는 저장 조건, 탈락 후 후속 연도 학습 생략, 탈락 파일 제거, 첫 전체 통과에서 정지, 재시작 시 재계산 방지를 테스트했다. 기존 계산 검증과 함께 16개 테스트를 통과했다.'),
        p(f"모든 worker 누적 실행 시간 {summary.get('elapsed_worker_seconds',0):.1f}초; worker 최대 RSS {summary.get('peak_rss_mib',0):.1f} MiB. 이 시간은 보고서 작성·Drive 업로드·프로세스 시작 등 전체 대기 시간을 포함하지 않는다."),
        t(['파일','내용'],[
            ['plan.json','전체 후보 설정·순서, 난수 시드, 범위, 기준, 데이터·수치 소스 해시'],
            ['all_attempts.jsonl / CSV','모든 시도의 설정 식별자, 상태, 탈락 사유, 연도, 사용 자원'],
            ['accepted_estimates.csv','통과 사례 추정치, 명목 p, 구간, 지지 진단 및 탐색 보정 p'],
            ['accepted/random_NNN/','통과 사례의 원본 설정, 연도별 상세 추정치·진단·그림'],
            ['SEARCH_SUMMARY_KO.md','실제 진행 수와 종료 사유'],
            ['docs/RANDOM_SEARCH_KO.md','전체 범위, 명령어, 보관 정책 및 해석 안내']], [1.7,3.5],9),
        p('재개 명령은 프로젝트 README와 실행 안내를 따른다. 같은 계획을 재개하며 이미 전체 목표를 찾았다면 추가 계산하지 않는다. 입력 파일이나 수치 계산 소스가 바뀌면 재개를 차단한다.'),
        h('해석의 제한'),
        p('부트스트랩은 학습된 nuisance, 선택한 지지 및 가중치에 조건부이며 학습·선택 전체를 재추정하지 않는다. 동시기 통제변수는 사후 변수일 수 있다. 통제 시점과 지지 변경으로 평균 대상이 바뀔 수 있다. 산업 노출은 정적이며 완전한 동적 처치 이력 모형이 아니다.'),
        p('Haddad 및 Callaway 논문은 기본 추정과 식별 논의의 참고 자료다. 구현이 논문과 완전히 동일하거나 요청한 패턴이 인과 식별을 보장한다고 주장하지 않는다. 기존 METHODS_KO.md의 한계도 함께 적용된다.'),
        p('원자료와 학습 모형 캐시는 결과 ZIP에 포함하지 않았다. Google Drive Labor_Productivity / Random_Search_200에 계획, 중간 체크포인트, 최종 결과와 보고서를 저장한다.')
    ])
    reported=[]
    for r in accepted:
        run=r['run_id'];folder=root/'accepted'/run
        config=json.loads((folder/'config.json').read_text());audit=json.loads((folder/'preprocessing.json').read_text())
        table=data.loc[data.run_id.eq(run)&data.effect.isin(EFFECTS)].copy()
        order={e:i for i,e in enumerate(EFFECTS)};table['order']=table.effect.map(order);table=table.sort_values(['year','order'])
        table_rows=[]
        for row in table.itertuples():
            table_rows.append([str(row.year),NAMES[row.effect],n(row.estimate),f'[{n(row.ci_lower)}, {n(row.ci_upper)}]',
                probability(row.p_value),probability(row.p_bonferroni_cumulative_budget)])
            reported.append({'run_id':run,'year':row.year,'effect':row.effect,'estimate':row.estimate,'p_value':row.p_value})
        pages.append([
            h(f'상세 사례 {run}: 연도별 주요 효과'),
            p('명목 95% 신뢰구간과 명목 p값이다. 마지막 열은 이전 64개 설정을 포함한 누적 탐색 Bonferroni p값이다. 표시 값은 반올림했으며 판정은 원래 값으로 했다.'),
            t(['연도','효과','추정치','명목 95% 구간','명목 p','누적 보정 p'],table_rows,[.5,.8,.8,1.9,.8,.9],8.8),
            p('전체 목표: '+str(r['pattern']['pattern_match'])+'; 우수 조건 포함: '+str(r['pattern']['preferred_match'])+'.'),
            p(f"실현 표본: 기업 {audit['analysis_firms']:,}개, 산업 {audit['analysis_industries']}개. Fold 요청 {config['n_splits']}, 반복 {config['n_repeats']}. 통제변수 시점: {config['covariate_time']}.")
        ])
        pages.append([
            h(f'상세 사례 {run}: 학습 변수·설정·그림'),
            p('GPS 변수: '+', '.join(config['gps_features'])+'.'),
            p('Mu 변수: '+', '.join(config['mu_features'])+'.'),
            p('GPS: '+config['gps_model']+f"; knot {config['gps_n_knots']}, 차수 {config['gps_degree']}. Mu: "+config['mu_model']+' '+json.dumps(config['mu_params'])+'.'),
            p(f"ESS {config['ess']}, 하한 ESS {config['min_ess_prime']}, 트리밍 {config['trim_threshold']}, 밀도 하한 {config['density_floor']}, 대역폭 배율 {config['bandwidth_scale']}, 차분 {config['diff_step']}, {config['gap_policy']}. 산업: {config['industry_scheme']}, 더미 {config['include_industry_dummies']}, 기준 {config['industry_min_groups']}/{config['industry_min_obs']}."),
            {'kind':'image','path':folder/'event_study.png','width':6.5,'height':5.3,
             'caption':'조건부 명목 95% 구간. 선택된 탐색 사례이며 독립 검증 결과가 아니다.'}
        ])
    stem='Random_Search_Report_KO';title='노동생산성 연속 DiD 무작위 변수·설정 탐색'
    renderers.word_report(pages,output/(stem+'.docx'),title)
    renderers.pdf_report(pages,stem,title)
    # Check that every retained primary estimate is represented exactly once.
    if accepted:
        expected=data.loc[data.effect.isin(EFFECTS),['run_id','year','effect','estimate','p_value']].sort_values(['run_id','year','effect']).reset_index(drop=True)
        observed=pd.DataFrame(reported).sort_values(['run_id','year','effect']).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,observed,check_dtype=False)
    manifest={'source_summary':summary,'report_pages_planned':len(pages),'verified_primary_rows':len(reported),
        'files':{path.name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in output.glob(stem+'.*') if path.suffix in ['.pdf','.docx']}}
    (output/'report_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT/'results/random200'))
    parser.add_argument('--output',default=str(ROOT/'work/random_reporting/output'));args=parser.parse_args()
    build(args.root,args.output)
