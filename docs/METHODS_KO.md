# 추정 방법과 논문과의 관계

## 참고한 판본

1. Callaway, Goodman-Bacon, Sant'Anna (2024), *Difference-in-Differences with a Continuous Treatment*, NBER Working Paper 32117, February 2024. 사용자가 첨부한 PDF의 판본이다. 연속 처치의 ATT·ACRT·ACR 구분, 처치수준별 선택 편의, 강한 평행추세 가정에 관한 논의를 참고했다. https://www.nber.org/papers/w32117
2. Haddad, Huber, Medina-Reyes, Zhang, *Difference-in-Differences with Time-varying Continuous Treatments Using Double/Debiased Machine Learning*. 첨부 PDF는 **arXiv:2410.21105v2, 2026-01-15** 판본이다. 2024년 최초 원고와 구분한다. 특히 §4의 패널 DR 표현 식 (19), §5의 cross-fitting·정규화 추정, §7의 Epanechnikov 커널·undersmoothing·10% IPW 트리밍을 참고했다. https://arxiv.org/abs/2410.21105

이 구현은 첨부 논문의 완전한 복제나 R `causalweight::didcontDML`의 포트가 아니다.
기존 노트북의 정규화 점수·표본 규칙을 출발점으로 만든 **DML형 연속 노출 DiD 탐색 도구**다.

## 현재 데이터가 허용하는 설계

업로드 데이터의 `office_ratio`는 산업 안에서, 그리고 2020–2025년 사이에 일정하다.
따라서 시간에 따라 실제로 달라지는 처치 이력을 사용한 Haddad 논문의 전체 동적 설계를 그대로 식별할 수는 없다.
여기서는 고정 산업 노출 D와 기준연도 2022 대비 결과 차이를 이용한다.
동일한 D를 과거 처치 이력으로 다시 조건화하면 조건부 처치의 변동이 사라질 수 있으므로 그렇게 하지 않는다.
서로 다른 실제 처치 이력 열이 있는 데이터에는 별도의 동적 설계가 필요하다.

기본 변환은 D = ln(100 × office_ratio), 결과는 ln_firm_prod다.
사후 ΔY = Y_year − Y_2022다. `pre_sign="legacy"`일 때 사전 ΔY는 기존 노트북과 같은
Y_2022 − Y_year다. `pre_sign="chronological"`은 사전에도 Y_year − Y_2022를 쓴다.
부호를 바꿔도 사전 양측 유의성은 달라지지 않지만 사전 점추정치의 방향은 달라진다.
2022의 그래프 0은 정규화 기준이며 별도로 추정하거나 사전 검정에 포함하지 않는다.

전체 요청 연도에 관측된 기업으로 균형패널을 만든다. 식별자·기업/산업 이동·처치 불변성을 검사한다.
선택한 공변량에 비유한 값이 있는 기업은 기업 단위로 제외하고 수를 기록한다.
규모는 2022년까지의 평균 매출을 산업 내 분위수와 비교해 정한다. 산업 기업 수가 기본 1개 이하면 규모 미분류다.
미분류 기업은 전체 효과에 포함되지만 SME·Large 효과에서는 제외된다.

## 학습

GPS는 산업별로 D의 조건부 평균과 분산을 학습한 정규 밀도다. 기본은 spline + ridge이며 선형 ridge도 선택할 수 있다.
평균과 분산은 각 outer training fold 안에서만 학습한다. 분산의 학습 목표는 training fold 내부의 OOF 평균잔차 제곱의 로그다.
테스트 기업의 D나 결과가 해당 기업의 GPS 학습 목표에 들어가지 않는다.
선형·스플라인 기반 정규 밀도는 실제 조건부 처치 분포가 정규라는 보장을 주지 않는다.

Mu는 ΔY를 공변량과 D로 예측하며 XGBoost, ridge, spline-ridge를 제공한다.
동일 산업의 기업은 반드시 같은 outer fold에 둔다. 실제 fold 수는 요청 수와 산업층별 산업 수의 최솟값이다.
각 연도의 반복 OOF 예측은 원본처럼 **중앙값**으로 합친다.
`tuning_scope="global_cv"`는 전체 표본의 산업 분리 CV로 Mu 설정을 한 번 선택하는 기존 방식에 가깝다.
이때 테스트 결과도 전역 하이퍼파라미터 선택에 간접적으로 관여한다. 더 엄격한 분리는
`tuning_scope="nested"`로 각 outer training fold 안에서만 Mu를 튜닝한다. 계산량은 늘어난다.

기본 GPS는 산업 공변량만, Mu는 기업 공변량·규모까지 사용한다. 서로 다른 조건화 집합이므로
Haddad 논문의 동일 X에 대한 DR 성질이 그대로 성립한다고 주장하지 않는다.
`configs/paper_inspired.json`은 GPS를 기업 수준으로 바꾸고 GPS·Mu에 동일한 공변량을 사용하며,
baseline 공변량과 nested 튜닝을 적용하는 예시다. 이것도 실제 식별 가정을 검증해 주지는 않는다.
`covariate_time="current"`인 동시점 임금·자산·R&D 등은 처치 후 변수일 수 있다.
`baseline`은 통제를 2022 값에 고정한다. 변수의 실제 측정시점과 인과적 역할은 연구자가 확인해야 한다.

## ATET와 ACR

커널 bandwidth b와 유한차분 간격 h=`diff_step`은 다른 값이다.
b는 산업별 D의 분산·IQR을 이용한 2.34 × scale(D) × G^(-1/4) × bandwidth_scale이다.
이는 논문의 undersmoothing 규칙을 산업별 처치 자료에 맞춘 선택이다. 독립 기업 수 대신 산업 수 G를 쓴다.

각 평가점 d에 대해 아래를 계산한다.

```text
r_i(d-h) = ΔY_i − Mu_i(d-h)
w_high_i = K_b(D_i-d)
w_low_i  = K_b(D_i-(d-h)) × GPS_i(d)/GPS_i(d-h)
a_i = w_high_i / sum(w_high)
b_i = w_low_i / sum(w_low)
ATET_hat(d,d-h) = sum(a_i r_i) − sum(b_i r_i)
ACR_hat(d) = ATET_hat(d,d-h) / h
```

두 가중평균은 각각 정확히 정규화한다. ATET를 h로 한 번 나눈 값이 여기서 저장하는 finite-difference ACR이다.
Callaway 논문의 전체 모집단 ACR과 자동으로 같아지는 것은 아니며, 처리집단 조건부 국소 대조에 가깝다.
`atet_estimate`, `atet_se`도 함께 저장한다. D 단위는 ln(100 × 비중)이며 비중 1%p당 효과가 아니다.
XGBoost는 구간별 상수 함수이므로 h를 지나치게 작게 하는 것만으로 매끄러운 도함수 추정이 보장되지 않는다.
차분 간격·bandwidth 민감도를 함께 점검해야 한다.

## 트리밍, ESS, 합산

현재 트리밍은 원본처럼 비교군 raw IPW의 **기업별 정규화 가중치**가 trim_threshold보다 큰 기업을 두 arm에서 뺀다.
그 뒤 두 arm의 가중치를 산업별로 합산한 ESS를 검사한다. 산업 대표행 압축 시에도 기업 수를 반영하므로 기업별 트리밍 기준이 유지된다.
이는 논문의 '모든 조건부 평균에서 큰 가중치 제거'를 완전히 복제한 규칙은 아니다.
GPS 밀도 바닥값과 trim_threshold는 서로 다른 설정이다.
`min_ess_prime`은 공통 처치 하한을 찾는 사전 필터이며, `ess`는 GPS 트리밍 뒤 각 국소 대조의 두 arm에 적용되는 기준이다.
후자의 지지는 ESS > ess로 판정한다. min_ess_prime만 낮췄다고 뒤 ESS를 생략하지 않는다.

처치 격자는 생략 없이 chunk 단위로 검사한다. ESS 지지점 사이에서 실제 산업의 D에 선형 보간하고 실제 기업 수로 평균한다.
`strict`는 누락된 격자점이 있거나 bandwidth보다 넓은 공백의 내부 기업을 제외하고 재정규화한다.
`bridge`는 공백을 연결하는 보간 가정을 허용한다. 각 집단의 포함 기업 수·coverage를 기록한다.
설정별 표본이나 support가 달라지면 추정 대상도 달라질 수 있으므로 유의성만으로 설정을 비교하지 않는다.

기본 조절효과는 Large의 자체 기업평균 − SME의 자체 기업평균이다. 두 집단의 support와 기업 가중치가 다를 수 있다.
추가로 공통 산업풀·전체 support·동일 보간 가중치로 비교한 `moderation_common`도 저장한다.
어느 경우든 조절효과의 음수는 Large의 ACR이 SME보다 작다는 뜻이다.

## 영향함수와 추론

정규화된 비율 추정의 영향함수는 다음과 같다.

```text
IF_i(d) = N × [a_i(r_i−mean_high) − b_i(r_i−mean_low)] / h
```

같은 산업 안의 IF를 합산한 뒤, 보간/기업 가중치로 **먼저** 합산하고 마지막에 정규 multiplier bootstrap을 적용한다.
이 합산은 nuisance 예측·support·최종 가중치를 고정하면 기존 점별 부트스트랩을 나중에 평균하는 것과 선형대수적으로 같다.
동일 multiplier를 SME·Large와 연도들에 공유해 차이 및 공동 검정의 공분산을 유지한다.
기본 cluster는 산업이다. 기업 cluster 옵션도 있으나 처치가 산업에 배정되어 있다는 점을 고려해야 한다.

각 효과에 nominal normal CI·양측 p-value와 multiplier percentile CI를 저장한다.
설정 안의 4효과×5연도에는 joint multiplier max-t 구간과 Holm p-value를 추가한다.
32개 탐색 전체의 사전 정의 family는 640개 검정이며, 미완료·실패 설정도 family 수에 포함해 Holm·Bonferroni 보정을 저장한다.
이 보정은 다중 비교를 다루지만 모델 선택·식별 가정·smoothing bias 문제를 모두 해결하지 않는다.
추론은 nuisance·선택된 support·보간 가중치를 고정한 조건부 추론이며 전 과정을 다시 학습하는 부트스트랩이 아니다.

2020·2021이 각각 유의하지 않다는 것은 평행추세의 증명이 아니다.
사전 8개 계수에 대한 joint Wald 검정도 진단으로 저장한다.
유의성 패턴을 보고 선택한 설정은 탐색 결과이며 독립 표본이나 사전 등록된 후속 분석의 확인 결과로 제시하면 안 된다.
