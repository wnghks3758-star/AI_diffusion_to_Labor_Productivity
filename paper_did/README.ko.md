# 논문에 맞춘 패널 ATET 구현 — 설계 협의 중

이 디렉터리는 기존 `run.py` / `labor_did` 추정 경로와 별개다.
참조는 첨부된 Haddad et al. arXiv:2410.21105v2와 causalweight 1.1.6의
`didcontDMLpanel.R` (84b6412)이다. 저자 패키지의 수치 선택과 산업 단위 처치에
필요한 조정을 구분해 기록한다. 현재는 수치 핵심만 구현되어 있으며 실행용
모형·자료 연결은 아래 선택을 받은 뒤 추가한다.

## 사용자와 정한 사항

- GPS는 `psmethod=1`: D의 조건부 평균을 예측한 후 정규밀도를 사용한다.
- 기업별 생산성 결과를 유지하고 GPS는 산업 단위로 학습한다.
- bandwidth는 산업별 한 행의 처치 D로 `2.34 * sd(D) * n^(-1/4) * 0.7`.
  `sd`는 R처럼 표본 표준편차(ddof=1), n은 산업 수이다.
- np Epanechnikov 커널: 실제 반폭은 `sqrt(5) * bandwidth`.
- 산업 전체를 같은 Fold에 배정하고 표준오차는 산업으로 군집화한다.
- ESS는 진단만 기록한다. ESS 기준으로 처치점·결과를 제외하지 않는다.
- 과거 유의성·효과 방향으로 설정을 탐색하거나 결과를 선별하지 않는다.

## 답변을 기다리는 선택

1. 학습기: 저자 기본 Lasso / RF / 기존 Spline·XGBoost.
2. 공변량: GPS·Mu의 공통 산업 변수 / 기존 산업 GPS와 기업·산업 Mu의 분리.
3. 이후 협의: 공변량 측정 시점과 결측 처리, ATET 검증 후 ACRT 연결 범위.

GPS와 Mu에 같은 공변량을 사용하는 것이 저자 R 함수의 구조이다. 기업 변수까지
Mu에 추가하는 기존 방식은 선택할 수 있지만 별도의 확장으로 기록해야 한다.
산업 단위 GPS·Fold·n 역시 저자 함수에 기업 행을 그대로 입력하는 것과 다르다.

## 수치 핵심에서 맞춘 부분

`numerics.py`는 처치점 한 쌍의 계산에 다음을 제공한다.

- `paper_bandwidth`: 표본 SD와 0.7 배율; robust sigma나 최소 bandwidth 대체 없음.
- `np_kernel`: R np의 Epanechnikov 척도.
- `reference_fold_variance`: 저자 방식인 외부 검증 Fold의 잔차제곱 평균.
  이 분산 구성은 검증 처치 D를 사용한다. 학습 Fold에서만 분산모형을 만드는
  기존 Python의 nested 이분산 모형과 다르며, 두 방식을 혼동하지 않는다.
- `gaussian_log_ratio`: 정규밀도 비율의 로그. 고정 밀도 하한 없이 안정적으로 계산.
- `atet_score`: 두 팔 정규화·양쪽 트리밍(기본 0.1)·저자의 centered score에 따른
  산업 군집 표준오차와 ESS 진단. 반복 예측 중앙값이나 ESS 통과 필터를 넣지 않는다.

검증 표본의 결과 Y를 모형 학습에 사용하지 않는다. 공통 분산의 검증 D 사용은
참조 R 함수의 특성을 재현하기 위한 것으로 명시한다.

## 현재 검증

```bash
python -m unittest discover -s tests -p test_paper_numerics.py -v
```

기존에 실제 R 4.5에서 공식 함수의 점수 블록을 실행해 저장한 4개 고정 nuisance
사례(`audit/results/score_r.csv`)와 ATET·트리밍 수·표준오차가 일치했다.
전체 R 패키지의 모형 학습이 동일하다는 검증은 아니며, 학습기 선택 후 별도로
검증하고 차이를 보고한다.
