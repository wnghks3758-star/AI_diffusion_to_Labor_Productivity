# 통과 사례 주변의 200개 국소 탐색과 통합 보고서

기존 모형·설정 32개, 변수 비교 32개, 무작위 800개를 검토한 뒤 추가 후보 200개를 고정했다. 2020·2021년 전체·SME·Large·조절효과 8개의 명목 p≥.05, 2023년 전체 추정치<0 및 p<.05인 사례 37개를 기준 설정 후보군으로 삼았다. 최초 64개에도 현재의 ‘2023년 전체만 하락’ 기준을 적용해 4개를 포함했다. 과거 결과와 과거의 더 엄격한 판정은 수정하지 않았다.

필수 조건 5개 충족 수, 전체·SME·Large의 비감소 회복 구간 수(각 2개, 총 6개), SME2025 양수 여부를 차례로 내림차순 정렬했다. 이후 SME2025 명목 p, 차수와 ID를 오름차순으로 정렬했다. 상위 10개에 동일한 후보 예산 20개씩을 배정했다. 이 순위는 원하는 결과에 가까운 정도를 이용하는 탐색이며 모형 타당성이나 인과 식별을 보증하지 않는다.

기준 설정은 var_06, random_660, random_632, random_315, random_234, random_720, random_342, var_32, random_212, random_130이다. 각 기준에서 한 항목 변경 10개와 두 항목 변경 10개를 만든다. 각 후보의 before/after는 `results/adaptive200/plan.json`에 명시되어 있다.

- 변수: GPS 산업 변수 13개 또는 Mu 산업·기업·규모 변수 18개에서 한 개 추가·제거. 최소 2개를 유지한다.
- Support: ESS와 하한 ESS ±1, 트리밍 ±.025, 밀도 하한 ±.001, 대역폭 ±.05. 명시적인 상하한을 적용한다.
- 모형: 종류를 유지하면서 GPS spline knot·차수 ±1, Mu 트리 수 ±40·깊이 ±1·학습률 ±.01·행/열 비율 ±.05 또는 Ridge alpha를 절반/두 배로 바꾼다.
- Fold: 산업 단위 fold 수와 반복을 2~5 안에서 ±1로 바꾼다.
- 산업: refined 기준 산업수·관측수 ±1 또는 산업 더미 전환. 비활성 설정은 후보에서 제외한다.

모형 종류, 처치·결과, 기준연도, 사전 차분 부호, 규모 분류, 통제 시점, 전체 표본 기준 변수, gap 정책, 차분 간격, alpha .05와 모형 seed42는 기준 설정과 같게 유지한다. 새 후보의 bootstrap은 1,999회다. 기존 var_06·var_32의 999회와 다른 점은 `bootstrap_normalized_from`으로 분리해 기록한다. 생성기 seed는 20261022이다.

후보 200개를 새 결과 계산 전에 고정하고, 이전 864개 및 새 후보 사이의 설정 중복을 제외했다. 기준 설정을 순환하며 실행하고 저장·자동 정지 조건은 기존과 같다. 필수 전체 목표를 찾으면 즉시 멈춘다. 2025년 SME 유의한 양수는 추가 우수 조건이며 필수 정지 조건이 아니다. 실패도 시도에 포함한다. 결과에 따라 후보를 대체하거나 범위를 다시 넓히지 않는다.

```bash
# plan.json이 이미 있으면 다시 만들지 말고 search로 재개한다.
.venv/bin/python -m dml_acr.adaptive_experiments --output results/adaptive200
.venv/bin/python -m dml_acr.random_experiments search --output results/adaptive200
.venv/bin/python scripts/validate_random_results.py --root results/adaptive200
.venv/bin/python -m dml_acr.random_history --root results/adaptive200
.venv/bin/python scripts/validate_random_history.py --root results/adaptive200
.venv/bin/python -m dml_acr.integrated_results --current results/adaptive200 --output results/integrated
python scripts/build_integrated_report.py --output work/integrated_reporting/output
python scripts/validate_integrated_results.py --report work/integrated_reporting/output
```

통합문서는 기존과 같이 통과 사례별로 ‘연도별 주요 효과’ 1쪽과 ‘학습 변수·설정·그림’ 1쪽을 제공한다. 미통과 사례의 상세 추정치는 싣지 않는다. 최초 두 차수의 현재 기준 미통과 결과는 원래 파일에 보존하되 통합 문서와 업로드용 통합 결과에서는 제외한다. 전체 후보와 시도 상태는 감사 장부에 남긴다.

이번 배치는 200×20=4,000개 검정, 누적 계획은 (32+32+800+200)×20=21,280개 검정이다. 조기 정지하더라도 계획 예산을 줄이지 않는다. 이전 통과 사례에도 최신 누적 Bonferroni 값을 재계산하되 원본 추정치와 과거 보정 값은 바꾸지 않는다. 보정이 적응적 선택을 취소하거나 평행추세·인과 식별을 입증하지는 않는다.

Google Drive의 `Labor_Productivity/Adaptive_Local_200`에 고정 계획, 실행 중 체크포인트, 최종 결과, 통합 PDF·Word와 소스를 저장한다. 원자료와 학습 캐시는 업로드 묶음에 넣지 않는다.
