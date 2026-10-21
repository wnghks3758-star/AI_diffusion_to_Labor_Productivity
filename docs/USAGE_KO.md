# Python 모듈과 실행 안내

프로젝트 최상위 폴더에서 `.venv/bin/python`을 사용한다. 새 환경은 `./setup.sh`로 설치한다.
노트북을 읽거나 실행하지 않아도 모든 분석을 Python에서 수행할 수 있다.

## 단계별 파일

| 모듈 | 역할 |
|---|---|
| config.py | 설정 검증·JSON 읽기/쓰기·설정 ID |
| preprocessing.py | 엑셀/캐시, 균형패널, 결측 검사, 규모 구분, 산업 더미, 연도 차분 |
| preprocessing_legacy.py | 원본에서 분리한 산업분류·기준연도 차분 수치 함수 |
| models.py | 산업 단위 Fold, GPS·Mu 학습, 학습 캐시, 디스크 예측 |
| support.py | chunk별 ESS 검사, strict/bridge support, 기업평균 보간 가중치 |
| support_legacy.py | 원본에서 분리한 가중치·트리밍·ESS 수치 함수 |
| score.py | 정규화 ATET 차이/h 및 비율 영향함수 |
| aggregation.py | 산업 영향함수 합산, 공동 multiplier, SE·CI·p-value |
| estimation.py | 연도별 단계 연결, SME/Large·조절효과 |
| visualization.py | 연도 그래프와 설정 전체 비교 그림 |
| pipeline.py | 단일 설정 실행, 연도별 체크포인트·진단·공동 추론 |
| criteria.py | 요청한 결과 패턴 판정, Holm 보정 |
| experiments.py | 고정 32개 계획, 순차 worker, 재시작·전체 결과 장부·ZIP |
| io.py | atomic JSON, 입력 해시, 버전 기록 |

## 빠른 사용

```bash
# 작은 연산 예산으로 전체 경로 점검
.venv/bin/python -m dml_acr run --config configs/smoke.json --output results/my_smoke

# JSON 파일을 편집하지 않고 설정 덮어쓰기 (여러 --set 사용 가능)
.venv/bin/python -m dml_acr run --config configs/smoke.json --set gps_model=linear --set ess=8 --set trim_threshold=0.05 --output results/custom

# 원본과 같은 큰 연산 예산. GPS 학습 절차는 개선된 Python 구현이다.
.venv/bin/python -m dml_acr run --config configs/full_budget.json --output results/full

# 공유 공변량·baseline 통제·nested Mu 튜닝 예시
.venv/bin/python -m dml_acr run --config configs/paper_inspired.json --output results/shared_controls

# 고정된 32개 중 미완료 설정부터 실행/재개
.venv/bin/python -m dml_acr search --output results/screen32

# 한 설정만 더 실행 / 실패 설정도 다시 시도
.venv/bin/python -m dml_acr search --output results/screen32 --limit 1
.venv/bin/python -m dml_acr search --output results/screen32 --retry-failed

# 재학습 없이 장부와 다중비교 보정 갱신
.venv/bin/python -m dml_acr summarize --output results/screen32
```

라이브러리에서는 다음처럼 쓴다.

```python
from dataclasses import replace
from dml_acr import AnalysisConfig, run_analysis

cfg = AnalysisConfig.load('configs/smoke.json')
cfg = replace(cfg, gps_model='linear', ess=8, trim_threshold=0.05,
              n_splits=3, industry_min_groups=6, label='my_spec')
estimates, status = run_analysis(cfg, 'results/my_spec')
```

## 변경할 수 있는 설정

| 설정 | 의미 |
|---|---|
| data_path, dose, outcome, dose_transform | 데이터·처치·결과·로그변환 |
| base_year, years, pre_sign | 기준연도·분석 연도·사전 차분 방향 |
| gps_features, mu_features | 학습에 들어가는 열 목록. dose는 Mu에 자동 추가 |
| sample_reference_features | 표본 고정용 complete-case 기준 열. 학습에는 자동 추가하지 않음 |
| gps_level | industry 또는 firm. 산업 수준에서는 산업 안에서 공변량이 일정해야 함 |
| covariate_time | current 또는 baseline |
| gps_model, gps_n_knots, gps_degree, ridge_alphas | GPS 비선형성·ridge 후보 |
| mu_model, mu_params, mu_search_space, mu_search_n_iter | Mu 모델·고정 설정·탐색 후보·탐색 횟수 |
| tuning_scope, n_splits, n_repeats, inner_splits, seed | 전역/nested 튜닝·산업 분리 Fold·반복·난수 |
| include_industry_dummies, industry_scheme | 더미 포함 여부, refined/major/none 분류 |
| industry_min_groups, industry_min_obs | refined 산업 병합·희귀군 생성 기준 |
| small_industry_max_firms, size_quantile | 규모 미분류 기준과 산업 내 매출 분위수 |
| ess, min_ess_prime | 사후 양 arm ESS 기준·사전 공통 하한 ESS |
| trim_threshold, density_floor | 기업별 IPW 트리밍·GPS 밀도 바닥값 |
| bandwidth_scale, min_bandwidth, diff_step, max_dose | bandwidth와 차분 격자 |
| gap_policy, size_difference | strict/bridge, 자체 평균차/공통 가중치 차이 |
| bootstrap_unit, n_bootstrap, alpha | cluster·반복수·유의수준 |
| model_jobs, model_threads | 병렬 학습 수·각 모델의 스레드 수 |
| support_batch_points, grid_chunk_points | ESS 검사 batch/chunk 크기 |
| prediction_batch_points, max_prediction_rows | 메모리상 중앙값 계산 크기·예측 행 한도 |
| bootstrap_batch_size, cache_dir | multiplier batch와 캐시 위치 |
| save_bootstrap_draws | 작은 연도별 요약 draw만 추가 저장 |

오타 난 설정명과 잘못된 범위는 조용히 무시하지 않고 오류로 처리한다.
`dataclasses.replace`나 JSON을 사용하면 새 설정을 빠르게 만들 수 있다.
다만 **기존 고정 탐색 계획을 결과에 맞춰 수정하지 않는다**. 새 탐색은 별도 폴더에 계획을 생성하고 변경 이유를 남긴다.

## 저장 파일

단일 설정 폴더: `config.json`, `estimates.csv`, `endpoints.csv`, `event_study.png`,
`audit.json`, `preprocessing.json`, `pattern_flags.json`, `status.json`, `progress.jsonl`.
각 연도가 끝나면 `estimates.partial.csv`와 `audit.partial.json`에 먼저 저장하며 정상 완료 뒤 최종 파일로 모은다.
실패하면 `status.json`과 worker.log에 오류를 남기고 다음 설정을 계속한다.

`results/screen32/`의 주요 파일:

- plan.json / configs/: 결과를 보기 전 확정한 모든 설정 및 데이터 해시
- all_specifications.csv: 미실행·실패를 포함한 32개 장부와 패턴별 플래그
- all_estimates.csv: 모든 완료 설정의 효과·연도별 추정치와 보정 p-value
- matching_specifications.csv: 기본 패턴 조건 모두 충족
- preferred_specifications.csv: 기본 조건과 SME 2025 유의한 양수 모두 충족
- summary.json / SEARCH_SUMMARY.md: 중간·최종 집계
- runs/spec_XX/: 설정별 원본 결과·그림·진단·로그
- checkpoints/spec_XX.zip: 해당 설정과 그 시점의 전체 장부
- specification_plot_2025.png: 성공한 모든 설정의 2025 결과 비교

매칭이 0개일 때도 빈 매칭 파일과 전체 결과를 저장한다. 설정을 결과에 맞춰 계속 바꾸거나 성공만 남기지 않는다.

Drive의 `spec_XX.zip`은 실행 중에 저장한 시점별 스냅샷이다.
최종 파일 `02_all_32_results_final.zip`과 최종 전체 CSV를 우선 사용한다.
`artifact_notes.json`에는 추정치 변경 없이 조절효과의 표본 진단 열을 정리한 내역도 남겼다.

## 메모리·속도 설계

1. 엑셀은 내용 해시를 기준으로 pickle 캐시해 반복 Excel 파싱을 피한다.
2. GPS·Mu 캐시는 입력 데이터·변수·모델·Fold·버전에 따라 구분한다. 결과·표본이 달라지면 재사용하지 않는다.
3. 산업 안에서 일정한 GPS·D는 기업 수를 가중한 산업 대표행으로 ESS를 검사한다. 다르면 기업행 계산으로 돌아간다.
4. 전체 격자는 검사하되 연도별 평균의 실제 보간에 쓰이는 끝점만 예측한다.
5. Mu 반복 예측은 memmap에 기록하고 작은 point batch로 정확한 중앙값을 구한다. 디스크 여유가 필요하다.
6. 같은 XGBoost 처치 분기를 가진 평가점은 예측을 공유한다. 모델 파일은 fold별로 하나씩 불러온다.
7. 점별 `bootstrap × grid` 행렬 대신 최종 평균의 산업 영향함수를 먼저 합산한다.
8. 설정 하나당 별도 프로세스로 실행해 모델·배열 메모리를 다음 설정에 누적하지 않는다.

`status.json`의 peak_rss_mib는 Linux worker 최대 RSS이며 디스크 캐시는 포함하지 않는다.
`audit.json`에는 예측 디스크 크기, 저장 끝점 수, 학습 캐시 적중 여부, 피한 dense bootstrap 배열 크기를 기록한다.
배열을 줄이는 수치 개선과 hist/ridge·반복수·격자 변경에 따른 통계적 차이는 구분해야 한다.
후자는 설정에 명시되어 있으며 원본 결과와 동일하다고 주장하지 않는다.

## 검증

```bash
.venv/bin/python -m unittest discover -s tests -v
```

영향함수의 수치 미분, 합산 전후 부트스트랩·집단 공분산 일치, 산업 ESS 압축,
strict gap, 산업 Fold 분리, tree 예측 공유·반복 중앙값, 잘못된 설정 및 패턴 판정을 검사한다.

동일 nuisance 예측하에서 원본 노트북의 연도별 평균·표준오차와의 일치도 별도로 검사할 수 있다.
노트북 로딩은 이 검증 스크립트에서만 사용하며 Python 분석 패키지 자체는 노트북에 의존하지 않는다.

```bash
.venv/bin/python scripts/validate_notebook_parity.py
```
