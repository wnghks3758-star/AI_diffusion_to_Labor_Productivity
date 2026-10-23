# 실행 단계와 설정 설명

`run_paper.py`는 이 브랜치의 신규 실행기이다. 기존 `run.py`는 1772 경로이며
새 방식의 결과와 구분한다. 참조는 첨부 Haddad et al. arXiv:2410.21105v2와
[causalweight 1.1.6의 didcontDMLpanel.R](https://github.com/cran/causalweight/blob/84b6412a13a2b7953e73ae2cacc8b696f7a6a18a/R/didcontDMLpanel.R)이다.

## 1. 데이터와 전처리

`pipeline.prepare()`는 기존 `labor_did/data.py`, `industry.py`의 균형 패널·기업 규모
분류·산업 더미 생성을 재사용한다. 새 추정에서는 기존 GPS나 ESS 지원 필터,
robust sigma bandwidth, 밀도 하한을 사용하지 않는다.

- 기업별 생산성 결과를 유지한다. 산업 평균 결과로 바꾸지 않는다.
- 기본 종속변수는 `ln_firm_prod`, 처치는 `ln(office_ratio * 100)`이다.
- `covariate_time="baseline"`: 선택한 산업·기업 통제변수를 2022년 값으로 고정한다.
- `pre_sign="legacy"`: 과거에는 `Y_2022-Y_과거`, 이후에는 `Y_이후-Y_2022`이다.
  과거 계수는 기준연도와의 위약 비교이며 실제 과거 처치의 인과효과로 단정하지 않는다.
- 기업 규모는 기존처럼 2022년까지의 기업 매출을 산업 내 비교해 고정한다.
  미분류 기업은 전체 결과에는 포함하고 SME·Large 비교에서는 제외한다.
- 산업 분류는 기본 refined, 최소 산업 수·관측수 모두 4이다.
- 결측은 각 학습 Fold의 중앙값으로 대체하고 결측 표시변수를 붙인다.
  대체·스케일링·Spline은 학습 자료에서 적합한 뒤 검증 자료에 적용한다.

`gps_features`, `mu_industry_features`, `mu_firm_features` 목록으로 변수를 추가·제거한다.
`include_firm_controls=false`이면 기업 통제변수와 규모 더미가 Mu에서 모두 빠진다.
규모별 결과를 계산하는 구분 자체는 유지된다. 산업 더미는 별도 설정에 따라 자동 추가된다.

## 2. 모델 학습과 GPS

`models.py`의 `fit_regression()`과 `fit_nuisance()`가 담당한다.

- GPS는 산업당 한 행으로 D를 학습하고 같은 산업 기업에 예측값을 공유한다.
- Mu는 기업별 결과 차분을 처치 D와 선택한 통제변수에 회귀한다.
- GPS와 Mu에 동일한 산업 단위 외부 Fold를 사용한다.
- 기본 외부 3-Fold, 내부 10-Fold, seed 1, 반복 중앙값 집계 없음.
  내부 튜닝도 산업을 분리한다. NumPy 분할은 R의 난수 순서와 같다고 주장하지 않는다.
- 기본 GPS는 연속변수 Spline + Ridge, Mu는 XGBoost이다.
- 두 모형 모두 Spline, RF, Lasso, Ridge, OLS, XGBoost로 변경할 수 있다.
- Spline/Ridge는 8개 alpha, Lasso는 30개 alpha를 학습 자료 내부의 산업 CV로 선택한다.
  `*_params`에 `alpha`를 고정하면 그 튜닝을 생략한다. Python Lasso는 glmnet과
  같은 L1 회귀 계열이지만 같은 lambda 경로·엔진·난수·예측값은 아니다.
- RF/XGBoost는 설정한 고정 파라미터로 학습한다. 효과의 부호·p값으로 튜닝하지 않는다.
- 기본 XGBoost는 300 trees, depth 3, learning rate 0.03, subsample/colsample 0.8,
  min child weight 7, CPU exact이다. 과거 탐색 결과의 연도별 최적값을 가져오지 않는다.

GPS는 `psmethod=1`이다. 평균 모형을 바꾸어도 정규분포 가정은 남는다.
분산은 저자 함수처럼 외부 **검증 산업**의 처치 잔차제곱 평균으로 Fold마다 하나를 사용한다.
이는 검증 D를 사용하는 저자 구현의 선택이며 학습 표본만 사용하는 분산모형과 다르다.
검증 결과 Y는 Mu 학습에 사용하지 않는다. 이분산 모형과 smearing은 사용하지 않는다.

## 3. ATET 계산

`numerics.py`가 제공하는 함수는 다음과 같다.

| 함수 | 역할 |
|---|---|
| `paper_bandwidth` | 산업 D의 표본 SD × 2.34 × n^(-1/4) × 0.7 |
| `np_kernel` | R np의 variance-one Epanechnikov, 실제 반폭 sqrt(5)h |
| `reference_fold_variance` | 외부 검증 산업의 잔차제곱 평균 |
| `gaussian_log_ratio` | 정규밀도비의 로그, 밀도 하한 없이 계산 |
| `atet_score` | 두 팔 정규화·양쪽 트리밍·ATET·군집 점수·ESS 진단 |

`estimation.pair_scores()`는 전체·SME·Large와 Large−SME를 계산한다.
조절효과의 표준오차는 두 집단의 군집 점수 차이를 이용해 공분산을 반영한다.
기본 트리밍은 **기업별** 정규화 가중치가 어느 팔에서든 0.1을 넘으면 양쪽에서 제외한다.
ESS는 어떤 값이든 진단에 기록할 뿐 결과를 선별하지 않는다.

기본 ATET는 산업 D의 25분위→75분위 비교이다. `pair_scale="value"`이면 직접 지정한
처치값을 사용한다. 단위는 현재 변환한 D 척도이며 원래 office_ratio와 혼동하지 않는다.
표준오차는 저자 centered score를 산업별 합산하고 G/(G−1)을 반영한다.
정규근사 구간을 기본으로 보고하고 공통 5,000회 산업 multiplier의 구간도 별도 저장한다.

## 4. ACRT 변환·보간·합산

`estimation.compute_year()`와 `support.py`가 담당한다. 이 부분은 R 패키지의 직접 출력
이후에 추가한 확장이다. `ATET(d,d-diff_step)/diff_step`을 계산한 뒤, 실제 기업의
처치값에서 보간하고 기업 수로 평균한다. 기본 `diff_step=0.0005`이다.

계산 가능 구간 판단은 **트리밍 후 양쪽 가중치 합이 양수인지**만 확인한다.
ESS > 5 같은 필터나 효과 부호·유의성 필터를 적용하지 않는다. 기업별 계산과 같은
결과가 되도록 산업별 기업 수와 가중치를 압축해 전체 격자를 검사한다.
전체 격자에 Mu를 예측하는 대신, 실제 평균에 필요한 보간 끝점에서만 예측한다.

- `gap_policy="interpolate"` **기본**: 내부의 계산 불가 구간은 양쪽 가능한 점으로 선형 보간.
- `gap_policy="exclude"`: 내부 결측 구간에 있는 기업을 평균 대상에서 빼고 재정규화.
- 양쪽 점이 없는 구간 바깥에서는 두 방식 모두 외삽하지 않는다.
- `acrt_support="strict"`는 진단용: 모든 원래 격자를 유지하고 필요한 끝점이 실패하면
  전체 합산을 미정의로 둔다. 첫 시험의 실패 기록도 이 방식으로 보존했다.

전체·SME·Large는 각 집단의 기업 수 분포로 평균한다.
`size_difference="means"` **기본**은 두 집단의 자체 평균 차이다.
`size_difference="common"`은 공통 노출 평균의 차이다. 공통 평균에는 양 집단이 있는
산업의 노출분포와 세 곡선 모두 계산 가능한 구간을 사용한다. 자체 평균의 차이는
기업 규모별 노출 구성 차이도 반영할 수 있으므로 동일 처치점에서의 효과 차이와 구분한다.

`coverage`, `n_target_firms`, `n_used_firms`, `n_gap_firms`와 구간별 계산 가능 여부를
저장한다. 포함률을 숨긴 채 전체 기업에 대한 평균이라고 해석하지 않는다.

## 5. 설정·저장·시각화

`config.py`는 잘못된 모델명·설정·처치쌍을 실행 전에 확인한다.
`pipeline.py`는 표본·설정·소스/데이터 해시, 연도별 체크포인트, CSV·PNG·HTML을 저장한다.
`run_paper.py --grid`는 명시한 조합의 결과와 실패를 모두 남긴다. 보고되는 Holm 보정은
실행 내 주 결과에 대한 것이며 모형 탐색 전체의 확증 추론을 보장하지 않는다.

| 설정 | 기본값 / 설명 |
|---|---|
| `gps_model`, `mu_model` | spline / xgboost |
| `gps_params`, `mu_params` | 모형별 파라미터 사전 |
| `include_firm_controls` | true; Mu의 기업 변수 사용 |
| `include_industry_dummies` | true |
| `industry_scheme` | refined / major / none |
| `industry_min_groups`, `industry_min_obs` | 4 / 4 |
| `n_splits`, `inner_splits`, `seed` | 3 / 10 / 1 |
| `bandwidth_factor`, `trim` | 0.7 / 0.1 |
| `spline_n_knots`, `spline_degree` | 5 / 3 |
| `atet_pairs`, `pair_scale` | [[0.25,0.75]] / quantile |
| `compute_acrt`, `diff_step` | true / 0.0005 |
| `gap_policy` | interpolate / exclude |
| `size_difference` | means / common |
| `n_bootstrap` | 5000 |

예제 설정은 루트의 `examples/paper_*.json`에 있다. 일부 항목만 지정한 설정 파일은
나머지를 `PaperConfig` 기본값으로 채운다. 기존 paper_settings.json의 수정값을
자동 상속하는 것은 아니다. 명령행 `--set`은 읽은 설정 파일에 덮어쓴다.

## 6. 검증 범위와 이론적 한계

자동 테스트 29개를 통과했다. 새 검증은 모형 6종, Fold와 전처리, 검증 Y 누출 방지,
공분산, 산업 압축과 기업 계산의 일치, 보간/제외와 외삽 금지, 체크포인트를 포함한다.
공식 R 점수 블록은 합성 4개 사례 및 실제 nuisance를 고정한 6개 사례와 비교했다.
실제 사례의 표준오차 비교는 `cluster=NULL`로 **iid SE**를 대조한 것이다. 실제 분석의
군집 SE는 군집 점수 테스트로 별도 확인했으며 전체 R 패키지의 학습·군집 모듈을
그대로 실행한 검증이라고 부르지 않는다.

저자와 다른 학습기·공변량, 산업 단위 배정과 기업 가중 목표의 연결, 변수의 인과적
역할은 별도 조건이 필요하다. ACRT의 유한차분·보간·합산 및 그 추론은 저자 ATET 정리의
자동적인 보장이 아니다. 이번 결과는 사용자가 정한 설정의 시험이며 유의한 패턴을
찾기 위한 모형 탐색 결과가 아니다.
