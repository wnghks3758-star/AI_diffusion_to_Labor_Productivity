"""Use the established two-page case format, restricted to gate-passing cases."""
from pathlib import Path
import argparse
import json
import os
import hashlib
import pandas as pd
import report_renderers as renderers
from build_random_report import p,h,t,n,probability,EFFECTS,NAMES

ROOT=Path(__file__).resolve().parents[1]


def build(root,output,current='results/adaptive200'):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    work=output.parent/'tex';work.mkdir(parents=True,exist_ok=True)
    for var,name in [('XDG_CACHE_HOME','cache'),('TEXMFVAR','texmf-var'),('TEXMFCONFIG','texmf-config')]:
        directory=work/name;directory.mkdir(exist_ok=True);os.environ[var]=str(directory)
    renderers.WORK=work;renderers.OUT=output
    summary=json.loads((root/'summary.json').read_text());current=Path(current).resolve()
    local_summary=json.loads((current/'summary.json').read_text());plan=json.loads((current/'plan.json').read_text())
    cases=[json.loads(line) for line in (root/'retained_cases.jsonl').read_text().splitlines() if line]
    data=pd.read_csv(root/'accepted_estimates.csv') if cases else pd.DataFrame()
    anchors=pd.read_csv(current/'selected_anchors.csv')
    pages=[[
        {'kind':'title','text':'노동생산성 연속 DiD\n전체 탐색 통합 보고서'},
        p('2026년 10월 21일 · 기존 문서 양식 유지 · 저장 조건 통과 사례만 상세 기술'),
        t(['누적 계획 / 실제 시도','통합 상세 사례','필수 전체 목표 / 우수 사례'],[[
            f"{summary['planned']} / {summary['attempted']}",str(summary['retained_cases']),
            f"{len(summary['nominal_pattern_matches'])} / {len(summary['nominal_preferred_matches'])}"]],size=10),
        h('이번 추가 탐색의 결과'),
        p(f"기존 통과 사례 37개 중 기준 설정 10개를 선정해 각 20개씩 이웃 후보를 만들었다. 새 후보 {local_summary['attempted']}/200개를 시도했고, 상세 보관 {local_summary['accepted_detailed_cases']}개, 탈락 {local_summary['rejected']}개, 실행 오류 {local_summary['failed']}개다. 종료 상태: {local_summary['stop_reason']}."),
        p('이번 전체 목표 통과 ID: '+(', '.join(local_summary['nominal_pattern_matches']) or '없음')+'.'),
        h('사용자가 요청한 저장 필터'),
        p('2020·2021년의 전체·SME·Large·조절효과 8개가 모두 유한하고 명목 p≥.05이며, 2023년 전체 효과가 음수이고 p<.05인 경우만 상세 기술한다. 초기 SME와 Large의 하락은 저장의 필수 조건이 아니다.'),
        p('통합 대상은 최초 모형·설정 32개, 변수 비교 32개, 무작위 800개와 이번 국소 탐색이다. 최초 64개는 현재 저장 기준으로 다시 판정했다. 기존의 더 엄격한 2023년 세 집단 하락 기준으로 작성한 과거 판정 파일은 수정하지 않았다.'),
        h('해석 범위'),
        p('모든 사례는 요청한 패턴을 찾는 탐색 과정에서 선택됐다. 사전 비유의는 평행추세의 입증이 아니며, 기존 결과를 보고 고른 이웃 설정의 통과도 독립적인 확인 증거가 아니다.'),
        p(f"주요 검정은 설정당 20개다. 누적 계획 {summary['planned']:,}개를 모두 포함한 {summary['family_tests']:,}개 검정의 Bonferroni p값을 상세 표에 제공한다. 보정은 결과에 따른 선택이나 식별 문제를 해결하지 않는다.")
    ],[
        h('1. 전체 탐색 이력과 통합 범위'),
        t(['탐색 차수','계획 / 시도','현재 저장 기준 통과','탈락 / 오류'],[
            [b['suite'],f"{b['planned']} / {b['attempted']}",str(b['current_gate_passed']),f"{b['rejected']} / {b['failed']}"]
            for b in summary['batches']],[1.7,1.1,1.3,1],9),
        p('최초 두 차수는 각각 32개 전체의 추정을 끝낸 비교 실험이다. 표의 탈락 0은 현재 저장 조건을 모두 통과했다는 뜻이 아니다. 현재 기준에 맞는 사례만 이번 상세 절에 포함했다.'),
        p('후속 탐색은 2020·2021·2023년 순서로 저장 조건을 검사하며 미통과 시 후속 계산을 생략했다. 상세 추정치와 그림은 삭제하고 후보 설정·처리 상태·탈락 사유만 감사 장부에 보관했다. 계산하지 않은 결과를 비유의로 간주하지 않는다.'),
        p('기존 무작위 800개 중 실행 오류 268개는 산업층 안에 산업이 1개뿐이어서 산업 단위 층화 교차적합이 성립하지 않은 경우다. 해당 후보를 유리한 fold나 다른 후보로 대체하지 않았다.'),
        h('공통 추정과 전체 목표'),
        p('결과변수 ln_firm_prod, 처치 log(100×office_ratio), 기준연도 2022년이다. 산업 단위로 교차적합한 GPS와 Mu를 이용해 정규화된 이중강건 평균 대비의 수치 미분 ACR을 구한다. 저장한 ATET는 ACR×차분 간격이다.'),
        p('전체 목표는 저장 조건에 더해 전체·SME·Large가 각각 2023≤2024≤2025 및 2025>2023을 만족하고, SME의 첫 양수 전환 연도가 Large보다 빨라야 한다. SME만 관측 기간 안에 양수로 전환해도 통과한다.'),
        p('조절효과는 Large 평균 ACR−SME 평균 ACR이다. 사후 세 연도 모두 음수이고, 세 연도 모두 유의하거나 2025년 유의 및 p2025≤p2023이어야 한다. 2025년 SME 유의한 양수는 추가 우수 조건이다. 필수 전체 목표를 찾으면 즉시 정지한다.')
    ],[
        h('2. 기준 설정 선정과 국소 변경'),
        p('새 계산 전에 후보 200개와 순서를 고정했다. 기존 저장 조건 통과 37개에서 필수 조건 충족 수(5개), 비감소 회복 구간 수(6개), SME2025 양수 여부 순으로 내림차순 정렬하고, SME2025 명목 p와 차수·ID를 오름차순으로 사용했다. 이는 추정 결과에 근거한 탐색용 순위이며 모형의 타당성 순위가 아니다.'),
        t(['순위','차수 / 기준 ID','필수 충족 / 5','회복 구간 / 6'],[
            [str(r.anchor_rank),f'{r.suite} / {r.run_id}',str(r.core_pass),str(r.recovery_transitions)] for r in anchors.itertuples()],
            [.5,2.8,1.2,1.2],9),
        p('각 기준 설정에서 한 항목 변경 10개와 두 항목 변경 10개를 생성했다. GPS 또는 Mu 변수 한 개 추가·제거, ESS±1, 트리밍±.025, 밀도 하한±.001, 대역폭±.05, GPS knot·차수±1, Mu 트리 수±40·깊이±1·학습률±.01·행/열 추출률±.05 또는 Ridge alpha 절반/두 배를 적용했다.'),
        p('Fold 수·반복은 2~5 안에서 ±1, refined 산업 기준은 ±1 또는 산업 더미를 전환한다. 비활성 설정은 바꾸지 않는다. 1개 변수 토글을 1항목으로 세며 서로 다른 두 Mu 매개변수는 2항목이다. 기존 전체 후보와 새 후보 사이 중복을 제외했다.'),
        p('기준 설정의 모형 종류·통제 시점·표본 기준 변수·gap 정책·차분 간격·결과·처치·규모 분류·모형 seed42·alpha .05는 유지했다. 새 후보는 bootstrap 1,999회로 통일했다. var_06·var_32 기준 설정의 과거 999회와 다른 계산 설정임을 별도로 기록했다. 생성기 seed=20261022, 기준 설정을 교차해 실행한다.')
    ],[
        h('3. 검증·재현·해석의 제한'),
        p('검증은 실제 추정 결과와 원본 표의 일치, 저장 필터·전체 목표 재판정, 원본 계획·데이터·수치 소스 해시 유지, 중복 설정 제외, 누적 보정, Word 표 숫자와 PDF 문자·페이지를 포함한다. 계산 및 탐색 관련 20개 단위 테스트를 통과했다.'),
        p(f"이번 worker 누적 실행 {local_summary.get('elapsed_worker_seconds',0):.1f}초, worker 최대 RSS {local_summary.get('peak_rss_mib',0):.1f} MiB. 보고서 작성·업로드·프로세스 시작을 포함한 총 대기 시간과는 다르다."),
        t(['파일','내용'],[
            ['all_specifications_audit.csv','모든 차수의 후보 상태와 현재 저장 기준 통과 여부'],
            ['plan.json','전체 설정과 과거 계획 파일의 해시'],
            ['accepted_estimates.csv','통과 사례만의 원본 추정치와 최신 누적 보정 p'],
            ['accepted / 후보 ID','통과 사례의 설정·전처리·추정치·그림'],
            ['adaptive200/plan.json','고정된 새 후보와 각 기준 설정 대비 변경 내역'],
            ['anchor_pool / selected_anchors','기준 설정 후보군·선정 순위·원본 해시']], [2.1,3.6],9),
        p('사전 차분 부호는 기존 코드의 2022년−사전 연도이며, 사후는 해당 연도−2022년이다. SME와 Large는 산업별 사전 평균 매출 중앙값으로 구분하며 법적 SME 정의가 아니다. 표본·지지·가중치가 다르면 평균 대상도 다를 수 있다.'),
        p('추론은 학습된 nuisance, 지지 및 가중치에 조건부인 산업 클러스터 multiplier bootstrap을 사용한다. nuisance·선택 전체를 다시 학습하지 않으며 동시기 통제는 사후 변수일 수 있다. 정적인 산업 노출을 쓰며 완전한 동적 처치 이력 모형은 아니다.'),
        p('Callaway 외(2024), Haddad 외(2024년 최초 원고·첨부 2026년 1월 판본)는 방법과 식별 논의의 참고다. 논문과 완전히 동일한 구현이라는 주장을 하지 않는다. 원자료·모형 캐시는 업로드용 결과 묶음에서 제외했다. 모든 미통과·오류는 상세 사례에서 제외하고 건수와 감사 기록으로 남긴다.')
    ]]
    # Keep the existing overview table without letting it spill across case pages.
    for start in range(0,len(cases),14):
        rows=[]
        for case in cases[start:start+14]:
            sub=data.loc[data.run_id.eq(case['run_id'])&data.year.eq(2025)].set_index('effect')
            rows.append([case['run_id'],f"{n(sub.loc['sme','estimate'])} ({probability(sub.loc['sme','p_value'])})",
                f"{n(sub.loc['moderation','estimate'])} ({probability(sub.loc['moderation','p_value'])})",
                '충족' if case['pattern']['pattern_match'] else '미충족'])
        pages.append([h(f'4. 저장 조건 통과 사례 {start+1}~{min(start+14,len(cases))} / {len(cases)}'),
            p('아래 p값은 명목 양측 p값이다. 최신 누적 보정 값과 과거·사후 전체 경로는 뒤의 사례별 표에 실었다.'),
            t(['후보 ID','2025 SME 추정치 (p)','2025 조절효과 (p)','전체 목표'],rows,[1.1,1.8,1.8,.9],8.7)])
    reported=[]
    local_entries={e['run_id']:e for e in plan['entries']}
    for case in cases:
        run=case['run_id'];folder=root/'accepted'/run;config=json.loads((folder/'config.json').read_text())
        audit=json.loads((folder/'preprocessing.json').read_text());table=data.loc[data.run_id.eq(run)&data.effect.isin(EFFECTS)].copy()
        table['order']=table.effect.map({e:i for i,e in enumerate(EFFECTS)});table=table.sort_values(['year','order'])
        rows=[]
        for row in table.itertuples():
            rows.append([str(row.year),NAMES[row.effect],n(row.estimate),f'[{n(row.ci_lower)}, {n(row.ci_upper)}]',
                probability(row.p_value),probability(row.p_bonferroni_cumulative_budget)])
            reported.append({'run_id':run,'year':row.year,'effect':row.effect,'estimate':row.estimate,'p_value':row.p_value})
        pages.append([h(f'상세 사례 {run}: 연도별 주요 효과'),p(f"출처: {case['suite']}. 명목 95% 신뢰구간·p값과 누적 계획 {summary['planned']}개 설정을 포함한 {summary['family_tests']:,}개 검정의 Bonferroni p값이다. 판정은 반올림 전 값으로 했다."),
            t(['연도','효과','추정치','명목 95% 구간','명목 p','누적 보정 p'],rows,[.5,.8,.8,1.9,.8,.9],8.8),
            p('전체 목표: '+str(case['pattern']['pattern_match'])+'; 우수 조건 포함: '+str(case['pattern']['preferred_match'])+'.'),
            p(f"실현 표본: 기업 {audit['analysis_firms']:,}개, 산업 {audit['analysis_industries']}개. Fold 요청 {config['n_splits']}, 반복 {config['n_repeats']}. 통제 시점: {config['covariate_time']}; bootstrap {config['n_bootstrap']}회.")])
        blocks=[h(f'상세 사례 {run}: 학습 변수·설정·그림')]
        if run in local_entries:
            entry=local_entries[run]
            blocks.append(p('기준 설정 '+entry['anchor_run_id']+'; 변경: '+'; '.join(
                c['coordinate']+': '+(('/'.join(sorted(set(c['before'])^set(c['after'])))) if isinstance(c['before'],list)
                    else str(c['before'])+' → '+str(c['after'])) for c in entry['changes'])+'.'))
        blocks.extend([
            p('GPS 변수: '+', '.join(config['gps_features'])+'.'),p('Mu 변수: '+', '.join(config['mu_features'])+'.'),
            p('GPS: '+config['gps_model']+f"; knot {config['gps_n_knots']}, 차수 {config['gps_degree']}. Mu: "+config['mu_model']+' '+json.dumps(config['mu_params'])+'.'),
            p(f"ESS {config['ess']}, 하한 ESS {config['min_ess_prime']}, 트리밍 {config['trim_threshold']}, 밀도 하한 {config['density_floor']}, 대역폭 {config['bandwidth_scale']}, 차분 {config['diff_step']}, {config['gap_policy']}. 산업 {config['industry_scheme']}, 더미 {config['include_industry_dummies']}, 기준 {config['industry_min_groups']}/{config['industry_min_obs']}.")])
        blocks.append({'kind':'image','path':folder/'event_study.png','width':6.5,'height':4.9 if run in local_entries else 5.3,
            'caption':'조건부 명목 95% 구간. 선택된 탐색 사례이며 독립 검증 결과가 아니다.'})
        pages.append(blocks)
    stem='Integrated_Search_Report_KO';title='노동생산성 연속 DiD 전체 탐색 통합 보고서'
    renderers.word_report(pages,output/(stem+'.docx'),title);renderers.pdf_report(pages,stem,title)
    expected=data.loc[data.effect.isin(EFFECTS),['run_id','year','effect','estimate','p_value']].sort_values(['run_id','year','effect']).reset_index(drop=True)
    observed=pd.DataFrame(reported).sort_values(['run_id','year','effect']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,observed,check_dtype=False)
    manifest={'summary':summary,'report_pages_planned':len(pages),'verified_primary_rows':len(reported),
        'case_ids':[r['run_id'] for r in cases],'files':{path.name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in output.glob(stem+'.*') if path.suffix in ['.pdf','.docx']}}
    (output/'report_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['summary','case_ids']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/integrated');parser.add_argument('--current',default='results/adaptive200')
    parser.add_argument('--output',default='work/integrated_reporting/output');args=parser.parse_args();build(args.root,args.output,args.current)
