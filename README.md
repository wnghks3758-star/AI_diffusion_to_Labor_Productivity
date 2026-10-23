# 기업 생산성 · 산업 단위 연속처치 DiD

**새 실행 명령은 `python run_paper.py`입니다.** 저자 ATET의 수치 계산에 맞추고,
기업별 결과와 산업 단위 처치 구조를 유지했습니다. 이 브랜치는
`feature/paper-aligned-did`이며 기존 실행기는 [1772 문서](README_1772.md)에 있습니다.

## 설치와 기본 실행

```bash
git clone --single-branch --branch feature/paper-aligned-did https://github.com/wnghks3758-star/AI_diffusion_to_Labor_Productivity.git
cd AI_diffusion_to_Labor_Productivity
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_paper.py --out results/my_trial
```

Windows의 가상환경 활성화 명령은 `.venv\Scripts\activate`입니다.
연속처치 데이터가 포함되어 있어 경로를 옮길 필요가 없습니다.

설정은 [paper_settings.json](paper_settings.json) 한 파일에서 변경합니다.
기본값은 GPS Spline·Mu XGBoost, 산업 GPS·기업 결과, 2022년 통제변수,
산업 3-Fold, SD bandwidth × 0.7, R np 커널·양쪽 트리밍입니다.
ESS는 진단만 남기고 밀도 하한은 적용하지 않습니다.

## 자주 바꾸는 설정

```bash
# GPS를 Random Forest로 변경: 밀도는 계속 조건부 정규밀도(psmethod=1)
python run_paper.py --set 'gps_model="random_forest"' --out results/gps_rf

# Mu를 Lasso로 변경
python run_paper.py --set 'mu_model="lasso"' --out results/mu_lasso

# Mu에서 기업 변수와 기업 규모 더미를 모두 제외: 규모별 분석 자체는 유지
python run_paper.py --set include_firm_controls=false --out results/no_firm

# 계산 불가 구간을 보간하는 기본값 대신, 그 구간의 기업을 제외
python run_paper.py --set 'gap_policy="exclude"' --out results/exclude

# ATET만 빠르게 시험
python run_paper.py --set compute_acrt=false --out results/atet_only

# 직접 지정한 처치값 비교: 값은 현재 변환된 D 척도
python run_paper.py --set 'pair_scale="value"' \
  --set 'atet_pairs=[[2.8,3.5]]' --out results/atet_pair

# 명시한 학습기 조합을 비교하고 모든 결과·실패 기록
python run_paper.py --grid 'gps_model=["spline","random_forest","lasso"]' \
  --set compute_acrt=false --out results/gps_comparison
```

두 모형 모두 `spline`, `random_forest`, `lasso`, `ridge`, `linear`, `xgboost`를 지원합니다.
모형별 파라미터는 `gps_params`, `mu_params`로 전달합니다. 예시:

```bash
python run_paper.py \
  --set 'gps_model="random_forest"' \
  --set 'gps_params={"n_estimators":300,"max_depth":3,"min_samples_leaf":8}' \
  --set 'mu_params={"n_estimators":500,"max_depth":3,"learning_rate":0.02}' \
  --out results/custom
```

## 결과와 재현

- `summary.csv`: 전체·SME·Large·조절효과의 ATET와 ACRT.
- `estimates.csv`: 공통 노출분포를 사용한 집계까지 포함한 전체 결과.
- `report.html`, `*.png`: 표·그림·실행 설정.
- `years/연도/`: 계산 가능 구간, 처치점별 결과, Fold·모형 진단, 군집 점수.
- `config.json`, `sample.json`, `status.json`: 전체 설정·표본·구현/자료 해시.

동일한 코드·설정·데이터로 다시 실행하면 완료 결과나 연도별 체크포인트를 재사용합니다.
다른 설정을 같은 출력 폴더에 쓰려 하면 중단하므로 새 `--out`을 지정하세요.

[단계별 코드와 설정 설명](paper_did/README.ko.md) ·
[실제 시험 보고서](audit/Paper_Aligned_Trial_ko.md) ·
[저자 ATET 구현 비교](audit/ATET_comparison_ko.md)

![기본 시험 ACRT](audit/figures/paper_acrt.png)

기본 시험은 기업 2,028개·산업 151개로 수행했습니다. ACRT 평균의 포함 범위는
계산 가능한 처치 구간의 기업 2,024개(99.80%)입니다. 미포함 기업과 실패도 기록했습니다.
저자 패키지의 전체 재현은 아닙니다. Spline·XGBoost 선택, GPS/Mu 공변량 차이,
산업 단위 학습·분할·표준오차, ACRT 변환·보간·합산은 구분해 설명합니다.

```bash
python -m unittest discover -s tests -v
```
