# DML ACR 노트북 실행 환경

## 모듈형 Python 분석과 32개 설정 탐색

노트북 없이 실행하는 `dml_acr/` 모듈을 추가했습니다.
전처리, GPS·Mu 학습, ESS·트리밍, ATET/ACR 합산, 시각화, 설정 탐색을 단계별로 분리했습니다.

```bash
.venv/bin/python -m dml_acr run --config configs/smoke.json --output results/my_run
.venv/bin/python -m dml_acr run --config configs/smoke.json --set gps_model=linear --set ess=8 --output results/my_linear
.venv/bin/python -m dml_acr search --output results/screen32
```

- [모듈 구조와 설정·실행 안내](docs/USAGE_KO.md)
- [추정식, 논문과의 관계, 추론과 해석](docs/METHODS_KO.md)
- [32개 설정의 실제 탐색 결과](docs/RESULTS_SCREEN32_KO.md)
- [전체 설정·조건별 판정](results/screen32/all_specifications.csv)
- [전체 추정치와 다중비교 보정](results/screen32/all_estimates.csv)

32개 모두 정상 완료했습니다. 사전 지정한 모든 패턴 조건을 만족한 설정은 0개입니다.
성공한 경우만 골라 남기지 않고 전체 설정·결과를 저장했습니다.
현재 결과는 반복 3회·bootstrap 999회의 1차 탐색이며 원본의 큰 연산 예산 결과와 구분합니다.
### 변수 추가·제외 비교

추가로 동일한 기업 표본과 모델·support 설정을 유지하는 32개 변수 설정을 비교했습니다.
GPS 변수 13개·Mu 변수 16개의 개별 제외와 두 변수군 구성을 포함합니다.

```bash
.venv/bin/python -m dml_acr.variable_experiments search --output results/variables32
```

- [변수 비교 설계·실행 방법](docs/VARIABLE_ANALYSIS_KO.md)
- [변수별 실제 결과 설명](docs/VARIABLE_RESULTS_KO.md)
- [2025년 기준 대비 변화](results/variables32/variable_comparisons_2025.csv)
- [변수 비교 전체 결과](results/variables32/all_estimates.csv)

이번에도 전체 요청 패턴을 모두 만족한 설정은 0개입니다. 기준 대비 추정치 변화는 기술적 비교입니다.

### 무작위 변수·설정 탐색

추가 요청에 따라 최대 200개의 무작위 후보를 고정하고, 사전 8개 효과 비유의와
2023년 전체 효과의 유의한 하락을 통과한 사례만 상세 저장하는 실행기를 추가했습니다.
전체 목표를 만족하면 즉시 멈추며, 탈락한 시도의 설정·사유는 감사 장부에 남깁니다.

```bash
.venv/bin/python -m dml_acr.random_experiments search --output results/random200
```

- [무작위 탐색 범위·저장 조건·자동 종료·재시작](docs/RANDOM_SEARCH_KO.md)
- [무작위 탐색 실제 결과와 통과 사례](docs/RANDOM_RESULTS_KO.md)
- [추가 200개: 후보 201~400 실행·재개·누적 보정](docs/RANDOM_CONTINUATION_KO.md)
- [후보 201~400의 실제 결과와 누적 결과](docs/RANDOM_RESULTS_201_400_KO.md)
- [추가 200개: 후보 401~600 실행·재개·누적 보정](docs/RANDOM_CONTINUATION_401_600_KO.md)
- [후보 401~600의 실제 결과와 누적 결과](docs/RANDOM_RESULTS_401_600_KO.md)
- [계속 실행: 후보 601~800 설계와 누적 보정](docs/RANDOM_CONTINUATION_601_800_KO.md)
- [후보 601~800의 실제 결과](docs/RANDOM_RESULTS_601_800_KO.md)
- [실행 현황과 종료 사유](results/random200/summary.json)
- [통과 사례만 포함하는 상세 결과](results/random200/accepted/)

아래는 기존 노트북을 실행하는 방법입니다.

프로젝트 위치: `/workspace/AI_diffusion_to_Labor_Productivity`

Python 3.12 가상환경(`.venv`), JupyterLab, 분석 패키지를 설치했습니다.
노트북과 엑셀 파일은 업로드 원본을 그대로 복사했습니다.

## 실행

```bash
cd /workspace/AI_diffusion_to_Labor_Productivity
./start_jupyter.sh
```

표시되는 JupyterLab 주소를 열고
`f4_7_DML_ACR_Dart_size_mean_difference_chunked.ipynb`를 여세요.
커널은 **DML ACR (Python 3.12)** 를 사용하세요.
첫 코드 셀(함수 정의)을 실행한 뒤 원하는 분석 셀을 실행하면 됩니다.
실행 스크립트가 프로젝트 폴더를 작업 디렉터리로 지정하므로 상대 데이터 경로가 맞습니다.
VS Code에서 실행할 때는 `.venv/bin/python`을 커널로 선택하고 작업 디렉터리를 이 폴더로 설정하세요.

## 데이터

```text
data/main_data/
├── firm_year_all_ind_dart_add_control_gpt_continuous.xlsx
├── firm_year_all_ind_dart_add_control_gpt_q50.xlsx
└── firm_year_all_ind_dart_add_control_gpt_q75.xlsx
```

현재 노트북의 `action_everything()`은 **continuous** 파일을 읽습니다.
q50·q75 파일도 같은 폴더에 배치했습니다. 이를 분석하려면 함수 내부의
`pd.read_excel(...)`에서 파일명을 선택하고 함수 정의 셀을 다시 실행하세요.

노트북의 분석 셀 세 개는 각각 아래 설정입니다.

| 분석 셀 순서 | diff_step | gap_policy | min_ess_prime |
|---|---:|---|---:|
| 1 | 0.0005 | bridge | 5 |
| 2 | 0.00001 | strict | 5 |
| 3 | 0.00001 | strict | 0 |

모든 셀 실행은 세 가지 분석을 차례로 수행합니다. 각 셀은 XGB 탐색 150개,
반복 50회, 부트스트랩 5,000회를 사용하므로 계산량이 큽니다.

## 설치 재현

Linux CPU 환경과 Python 3.12 기준입니다. 프로젝트를 옮긴 경우:

```bash
./setup.sh
./start_jupyter.sh
```

`requirements.in`은 직접 의존성, `requirements.lock.txt`는 설치한 전체 패키지의 고정 버전입니다.
XGBoost는 GPU 의존성이 없는 `xgboost-cpu==3.1.3`을 사용하며,
노트북의 `from xgboost import XGBRegressor`와 `tree_method='exact'`를 지원합니다.
NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0,
matplotlib 3.10.8, openpyxl 3.1.5를 사용합니다.
업로드 노트북에 저장된 과거 실행의 버전 및 검증 메타데이터는 이번 환경의 실행 기록과 구분해야 합니다.

## 환경 점검

```bash
.venv/bin/python scripts/validate_environment.py
```

원본 노트북의 첫 코드 셀을 실제 Jupyter 커널에서 실행하고,
엑셀 3개의 로딩·필수 열·기업-연도 중복·처치변수의 유효성을 검사합니다.
continuous 데이터 전체 표본으로 bridge와 strict를 축소 실행하여
학습, 교차적합, 부트스트랩, 연도별 요약, 그래프 생성을 점검합니다.

축소 점검 설정은 2 folds, 반복 1회, 부트스트랩 49회, diff_step=0.05,
XGB 탐색 2개 및 트리 8/12개입니다. 점검용 변경은 별도 커널 메모리에만 적용됩니다.
원본 전체 설정 분석은 이번 환경 구축에서 실행하지 않았습니다.

점검 결과는 `validation/environment_report.json`,
JupyterLab 서버 기동 점검 결과는 `validation/jupyter_server_report.json`에 있습니다.
임시 실행 노트북과 캐시는 `work/`에 저장합니다.
