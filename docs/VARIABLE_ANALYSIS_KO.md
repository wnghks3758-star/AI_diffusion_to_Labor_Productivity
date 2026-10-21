# 변수 추가·제외 비교

이번 실험은 `results/variables32/`에 별도로 저장한다. 기존 모델·support 탐색의 32개 결과는 보존한다.

## 32개 설계

| 설정 | 변경 |
|---|---|
| var_01 | 전체 공변량 기준 설정 |
| var_02–14 | GPS의 산업 공변량 13개를 하나씩 제외. Mu 변수는 고정 |
| var_15–27 | Mu의 산업 공변량 13개를 하나씩 제외. GPS 변수는 고정 |
| var_28–30 | Mu의 기업 R&D·임금·자산 변수를 하나씩 제외 |
| var_31 | GPS는 산업 자산·특허 2개만 포함. Mu는 이 2개와 기업 3개·규모 더미 포함 |
| var_32 | var_31에 사전 노동구성 공변량 9개를 GPS·Mu 양쪽에 추가 |

산업 공변량은 산업 평균임금, 자산, R&D, 누적 특허 및 사전 대졸 비중·대규모 사업체 비중·평균연령·50세 이상 비중·여성 비중·상용직 비중·평균근속·평균시간·평균임금이다.
산업 더미와 기업 규모 더미 생성 규칙은 모든 설정에서 같다. 한 단계에서 뺀 변수는 다른 학습 단계에 남아 있을 수 있으며 설정 이름에 GPS/Mu를 명시했다.
var_31→32는 동일 기본군에 노동구성 9개를 추가하는 중첩 비교다.

## 고정하는 것

- 전체 분석 공변량 16개의 공통 complete cases를 사용한다. `sample_reference_features`는 표본 제한용 열이며 자동으로 학습 변수에 추가되지 않는다.
- 모든 설정에서 같은 기업 1,991개, 산업 141개를 사용하도록 실행 전 검사했다. 실행 뒤에도 기업 집합 해시를 비교한다.
- 산업 분리 Fold 3, 반복 3회, seed 42, bootstrap 999회, diff_step=.002다.
- ESS 5, 공통 하한 ESS 5, 트리밍 .10, 밀도 바닥값 .005, bandwidth 배율 .5, strict 공백 처리를 고정한다.
- GPS 구조는 spline(knots 3, degree 3) + ridge다. ridge alpha 후보도 같다. 각 training fold에서의 alpha 학습은 변수 변화에 반응할 수 있다.
- Mu는 XGBoost histogram, 트리 160개, 깊이 3, 학습률 .05, 행/열 subsample .8이다. Mu 후보 탐색을 꺼서 모델 파라미터를 고정한다.

이 기준 설정은 앞선 32개 탐색의 튜닝된 spec_01과 구분한다. 이번 suite 안에서는 var_01을 비교 기준으로 쓴다.
대상 기업은 같아도 GPS가 바뀌면 실제로 ESS·트리밍을 통과하는 support와 평균 coverage는 바뀔 수 있다. 이 차이도 함께 저장한다.

## 결과 파일

- `plan.json`, `configs/`: 결과를 보기 전에 확정한 설정
- `all_estimates.csv`: 주요 계열과 추가 계열의 모든 연도 추정치
- `all_specifications.csv`: 전체 설정과 패턴 플래그
- `variable_comparisons.csv`: 같은 연도·효과에서 var_01 대비 추정치와 coverage 변화
- `variable_comparisons_2025.csv`: 2025년 주요 효과 비교
- `fixed_sample_audit.json`: 모든 완료 설정의 기업 집합 해시·표본 수
- `cumulative_64spec_primary_estimates.csv`: 앞선 탐색과 이번 탐색의 주요 1,280개 검정을 함께 보정
- `runs/var_XX/`: 설정별 전체 결과·그래프·진단
- `checkpoints/var_XX.zip`: 설정 완료 시점의 중간 스냅샷

추정치 변화는 **기술적 비교**다. 변수의 인과효과나 설정 간 차이에 대한 유의성 검정으로 해석하지 않는다.
각 설정의 p값과 두 설정 사이 차이의 p값은 서로 다르다. 이 실험에서는 후자의 p값을 계산하지 않는다.
이번 640개 주요 검정의 Holm/Bonferroni 보정과, 앞선 탐색을 포함한 1,280개 검정의 누적 Holm 보정을 구분한다.
요청한 패턴 플래그는 기존 정의를 그대로 사용하며 결과에 따라 기준을 바꾸지 않는다.

## 실행과 사용자 설정

```bash
# 별도 폴더에 계획 생성
.venv/bin/python -m dml_acr.variable_experiments plan --output results/variables32

# 실행·재개
.venv/bin/python -m dml_acr.variable_experiments search --output results/variables32

# 기존 결과에서 기준 대비 변화와 누적 보정만 다시 계산
.venv/bin/python -m dml_acr.variable_experiments summarize --output results/variables32
```

```python
from dataclasses import replace
from dml_acr import AnalysisConfig, run_analysis

cfg = AnalysisConfig.load('results/variables32/configs/var_01.json')
cfg = replace(cfg,
    gps_features=[v for v in cfg.gps_features if v != 'ln_ind_avg_pay'],
    label='my_gps_without_industry_pay')
run_analysis(cfg, 'results/my_variable_spec')
```

`sample_reference_features`를 유지하면 변수를 빼더라도 원래 변수의 결측 때문에 제외한 기업이 다시 들어오지 않는다.
이를 빈 목록으로 바꾸면 현재 학습 변수에 따른 표본을 사용한다. 두 표본 규칙은 목적이 다르므로 별도 실험으로 기록해야 한다.
