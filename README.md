# PS1 · 관측 처치점에서 직접 계산하는 ACRT

브랜치: `feature/ps1-observed-acrt`

실행 명령: `python run_observed.py`

기본 설정: 1538번 — GPS·Mu XGBoost, bandwidth 배율 0.5, trim 0.01, 차분 간격 0.0005.

각 산업의 실제 처치값 D에서 ATET(D, D−차분간격)를 구해 ACRT로 변환하고,
기업 수로 가중해 전체·SME·Large 및 Large−SME를 계산합니다.
관측점 사이의 보간은 사용하지 않습니다. 이전 분석 결과 파일을 내려받을 필요 없이
이 브랜치의 코드·설정·포함된 연속처치 데이터만으로 실행합니다.

## 설치와 실행

Python 3.12를 사용하세요. macOS/Linux:

```bash
git clone --single-branch --branch feature/ps1-observed-acrt https://github.com/wnghks3758-star/AI_diffusion_to_Labor_Productivity.git labor-productivity-ps1-observed
cd labor-productivity-ps1-observed
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_observed.py
```

Windows PowerShell에서는 위 clone·cd 후 다음을 실행합니다. 활성화 스크립트 없이도 됩니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_observed.py
```

Windows/Linux는 CPU 전용 XGBoost, macOS는 표준 XGBoost 패키지를 설치합니다.
모형 계산은 CPU로 수행하며 실제 설치된 배포판과 버전을 기록합니다.
macOS에서 OpenMP 라이브러리 오류가 발생하면 `brew install libomp` 후 다시 실행하세요.

## 설정 변경

[observed_settings.json](observed_settings.json)을 수정하면 됩니다.
GPS와 Mu의 산업 변수 목록은 같은 순서로 함께 수정합니다.
기업 변수는 Mu의 `mu_firm_features`에서 개별 포함·제외할 수 있습니다.

| 설정 | 기본값 / 의미 |
| --- | --- |
| `gps_model`, `mu_model` | 각각 `xgboost`; spline, random_forest, lasso, ridge, linear도 지원 |
| `gps_params`, `mu_params` | 각각 별도 학습기 파라미터 |
| `gps_features`, `mu_industry_features` | 동일한 산업 변수 목록 |
| `mu_firm_features`, `include_firm_controls` | 기업 변수 목록 / 전체 사용 여부 |
| `gps_n_splits`, `gps_fold_seed` | 4 / 107 |
| `mu_n_splits`, `mu_fold_seed` | 3 / 73 |
| `bandwidth_factor`, `trim` | 0.5 / 0.01 |
| `diff_step` | 0.0005; 직접 평가에서는 차분 폭만 결정 |
| `size_variable`, `size_threshold` | sale / median |
| `size_years`, `size_time_aggregation` | 기준연도 이하 패널 기간 / mean |
| `years`, `base_year` | 분석연도 목록 / 2022 |
| `model_threads` | 학습기 CPU 스레드 수 1 |

GPS 학습기를 바꾸어도 이 실행기의 GPS는 PS1의 조건부 정규밀도입니다.
Spline 변환은 모델을 spline으로 선택할 때 적용합니다.
분석연도 변경은 균형 표본과 사용 가능한 사전 규모 기간도 바꿀 수 있습니다.
기본 실행은 한 번의 산업 분할이며, 반복 분할 50회 집계는 이 명령에 포함되지 않습니다.

파일 수정 없이 숫자 설정을 덮어쓸 수도 있습니다. 설정마다 다른 출력 경로를 쓰세요.

```bash
python run_observed.py --set diff_step=0.00001 --out results/smaller_step
python run_observed.py --set gps_n_splits=5 --set mu_n_splits=3 --out results/folds_5_3
python run_observed.py --config my_settings.json --out results/my_trial
```

## 계산 가능 범위

기본 `--support supported`는 각 집단에서 계산 가능한 관측 처치점에 속한 기업을
평균 대상으로 삼고 그 비율을 coverage로 기록합니다. 이 값은 전체 기업 효과와
구분해야 합니다. 실패 처치점을 보간하거나 밀도 하한으로 대체하지 않습니다.

`--support strict`는 원래 모든 기업의 처치점이 계산 가능해야 집계를 제공합니다.
필요한 한 점이라도 실패하면 해당 효과는 failed·NaN이며 종료 코드는 2입니다.
모든 실패 지점과 두 방식의 집계는 기본 실행에서도 함께 보존합니다.
유의성으로 결과를 선별하거나 실행을 중단하지 않습니다.

```bash
python run_observed.py --support strict --out results/strict
```

## 결과 확인

기본 결과 폴더는 `results/ps1_observed/`입니다.

- `report.html`: 브라우저로 여는 결과 표·그래프·설정.
- `summary.csv`: 선택한 support 정책의 연도별 전체·SME·Large·조절효과, SE, p값, coverage.
- `estimates.csv`: supported와 strict 집계 모두.
- `atet.csv`: 설정한 고정 처치쌍 ATET.
- `years/연도/endpoints.csv`, `availability.csv`: 관측점별 효과·실패 사유·ESS·포함 기업 수.
- `years/연도/cluster_scores.npz`: 산업 기여도; 합산 후 SE와 규모 차이의 공분산 계산에 사용.
- `config.json`, `sample.json`, `audit.json`, `status.json`: 전체 설정·표본·산업 Fold·버전·자료/코드 해시.

같은 자료·설정·코드로 같은 폴더에 다시 실행하면 완료 결과와 연도 체크포인트를
재사용합니다. 다른 설정을 같은 폴더에 쓰면 중단합니다.
학습 모델 캐시는 `.cache/ps1_observed/`에 보관합니다.

p값과 구간은 보정 전 점별 추론입니다. 기존 산업 단위 GPS·산업 검증 Fold·산업 군집
기여도 합산 방식을 유지합니다. GPS와 Mu의 공변량 집합 차이, 산업 단위 처치에 대한
식별, 평활 편향·트리밍·ACRT 변환은 적용상의 검토 사항입니다.
`gap_policy`·균등 격자 설정·보조 bootstrap 설정은 이 직접 계산 경로에 사용하지 않습니다.

## 코드 구성과 검증

- [run_observed.py](run_observed.py): 설정·명령행·결과 출력.
- [ps1_observed.py](ps1_observed.py): 관측점 계산·연도별 저장·HTML/그래프.
- [observed_acrt.py](observed_acrt.py): 기업 수 가중치·관측점 합산·산업 점수.
- [paper_did/models.py](paper_did/models.py): 산업 단위 GPS/Mu 분할과 학습.
- [paper_did/numerics.py](paper_did/numerics.py): SD bandwidth·√5 커널·ATET 점수.

```bash
python -m unittest discover -s tests -v
```

실행 코드와 설정을 이 브랜치에 제공하며 새 분석 산출물·학습 모델은 Git에 포함하지 않습니다.
