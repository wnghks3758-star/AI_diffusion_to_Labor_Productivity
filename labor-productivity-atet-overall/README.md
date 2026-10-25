# PS1/PS2 최소 기준점 대비 전체 ATET

`labor-productivity-ps1-observed`에서 필요한 실행 코드와 원자료만 가져온 독립 패키지입니다. 기존 폴더의 코드·데이터·캐시를 읽지 않습니다. Python 환경은 기존 `../ACR`을 사용할 수 있으며, 다른 위치에서는 requirements.txt로 별도 설치하면 됩니다.

## 실행

Windows에서는 `run_atet.cmd`를 더블클릭합니다. 터미널에서는 다음과 같이 실행합니다.

```powershell
..\ACR\Scripts\python.exe run_atet.py
```

다른 작업 디렉터리에서도 `run_atet.py`의 전체 경로를 지정하면 실행됩니다. 기본 설정·원자료·모델 파라미터 경로는 패키지 폴더 기준입니다. 새 Python 환경을 사용할 경우:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_atet.py
```

실행마다 설정·자료·코드 해시에 따른 결과 폴더를 자동 선택합니다. 설정을 바꿔도 이전 결과를 덮어쓰지 않습니다. 동일 설정으로 다시 실행하면 완료 결과나 중간 체크포인트·모델 캐시를 재사용합니다. `--out 경로`로 결과 폴더를 직접 지정할 수도 있습니다.

## 기준점과 추정 대상

두 기준점을 함께 실행하는 `atet_settings.json` 설정 예입니다. 실제 실행값은 해당 파일에서 확인합니다.

```json
{
  "reference_doses": [0.153, 0.175],
  "sample_exclusion": "none",
  "support_policy": "strict"
}
```

`reference_doses`는 **원래 office_ratio의 단위**입니다. 15.3%는 `0.153`으로 입력합니다. `dose_transform="log_percent"`이면 내부에서 각각 `log(15.3)`, `log(17.5)`로 변환합니다. 15.3, 17.5를 입력하지 마십시오.

각 기준점 d0에 대해:

1. 실제 관측 처치강도 d > d0마다 ATET(d,d0)를 계산합니다.
2. 기준점 초과 기업의 처치강도별 비중 w(d)=N(d)/N(D>d0)으로 평균합니다.
3. ATET^o(d0)=sum_d w(d) ATET(d,d0)를 연도·전체·SME·Large별로 출력합니다. moderation은 Large−SME입니다.

대상은 E[Y_t(D)−Y_t(d0) | D>d0]입니다. 15.3%와 17.5%는 대상 기업 집단과 반사실이 다릅니다. 기준점 이하 기업은 집계 가중치를 받지 않지만, none 정책에서는 학습·비교군에 사용됩니다. Callaway식 처치분포 가중 평균을 목표로 하되, 개별 ATET는 기존 패키지의 Haddad·Huber 기반 PS1/PS2 DML 계산입니다. 저자의 R 추정기를 그대로 이식한 것은 아닙니다. 최소처치 상태의 조건부 평행추세와 공통지지 가정이 별도로 필요합니다.

## 기준점 아래 관측치 제외

`sample_exclusion`은 다음 세 값 중 하나입니다.

| 값 | GPS·Mu 학습 | ATET 비교군·가중치 계산 | 대역폭 |
|---|---|---|---|
| `none` | 전체 산업 | 전체 표본 | 전체 산업 처치강도로 계산 |
| `training` | 해당 기준점 이상 산업만 | 전체 표본 유지 | 전체 산업 처치강도로 계산 |
| `analysis` | 해당 기준점 이상 산업만 | 해당 기준점 이상 표본만 | 남은 분석 산업으로 재계산 |

`training`은 산업 단위 평가 폴드를 유지하면서 각 폴드의 학습 산업을 필터링합니다. GPS의 검증 잔차분산에도 기준점 아래 산업을 사용하지 않습니다. 그 산업의 OOF 예측은 학습된 모형으로 계산할 수 있으므로 비교군 계산에는 남습니다. 결측치 대체·표준화·하이퍼파라미터 선택도 필터링된 학습자료 안에서만 수행합니다. 폴드별 산업이 부족하면 실패 이유를 표시합니다.

`analysis`는 균형패널·기준 공변량·기업규모를 준비한 후 기준점 아래 산업을 분석에서 제거합니다. 기업규모 분류를 자의적으로 다시 바꾸지 않습니다. 두 제외 정책 모두 각 기준점별로 다른 모형을 학습하고 캐시를 분리합니다. `none`에서는 두 기준점이 같은 학습 모형을 공유합니다.

예:

```powershell
..\ACR\Scripts\python.exe run_atet.py --set 'sample_exclusion="training"'
..\ACR\Scripts\python.exe run_atet.py --set 'sample_exclusion="analysis"'
```

Windows PowerShell 버전에 따라 JSON 인자 따옴표 처리가 다르면 `atet_settings.json`의 값을 직접 수정하십시오. 제외 옵션을 켜면 앞선 15.3%·17.5%의 trimming 진단과 다른 결과가 나올 수 있습니다. 주어진 두 기준점을 유지한 채 정책을 바꿔 비교할 수 있습니다.

## 모델과 설정

- 모델 선택은 `atet_settings.json`의 `gps_model`, `mu_model`에서 합니다.
- 모델별 하이퍼파라미터는 `model_parameters.json`에 있습니다. GPS는 `gps.ps1` 또는 `gps.ps2`에서 해당 모델의 파라미터를 선택합니다. Mu 파라미터는 공통입니다. spline, lasso, ridge, linear, random_forest, xgboost를 지원합니다.
- GPS·Mu 모델, 산업별 분할 수, 반복 횟수, trim, 대역폭 배율은 `atet_settings.json`에서 지정합니다. `prediction_aggregation="median"`이면 반복한 OOF 예측의 중앙값을 사용합니다.
- `trim`은 기업별 정규화 가중치의 상한입니다. 산업 가중치 1% 상한이나 GPS 하한이 아닙니다. ESS는 진단으로만 출력하고 자동 제거하지 않습니다.
- `pre_sign="legacy"`이면 2022년 이전은 Y_2022−Y_pre, 이후는 Y_post−Y_2022입니다. `chronological`이면 이전도 Y_pre−Y_2022로 표시합니다.
- `psmethod=1`은 Gaussian GPS, `psmethod=2`는 공변량으로 커널 가중치를 예측하는 GPS입니다. `atet_settings.json`의 값에 따라 선택됩니다. PS2에는 Gaussian 처치 회귀나 잔차분산 계산을 사용하지 않습니다.
- PS2는 관측 처치강도와 기준점별로 커널 가중치 회귀를 따로 학습하므로 PS1보다 오래 걸릴 수 있습니다. 반복별 OOF 예측을 캐시하고 중앙값을 계산합니다. `training`에서는 기준점 이상 산업만 학습하고, 학습용 커널 정규화에도 그 산업만 사용합니다.
- PS2에서 필요한 커널 지지 구간의 예측값이 0 이하이면 해당 비교를 failed로 기록합니다. 임의의 양수로 치환하지 않습니다. 따라서 PS1에서 가능했던 15.3%·17.5%가 PS2에서도 모두 계산 가능하다는 뜻은 아닙니다. strict/supported 결과와 coverage를 확인하십시오.
- ACR, 차분 간격 및 과거 탐색 결과·캐시는 포함하지 않습니다. 결과 폴더·CSV·audit에 실제 PS 방식이 기록됩니다.

## 계산 실패와 추론

`strict`는 기준점 초과의 모든 관측 처치강도를 집계합니다. 하나라도 실패하면 해당 집계가 failed입니다. `supported`는 계산 가능한 지점만으로 가중치를 다시 정규화하며, coverage와 함께 다른 추정 대상으로 보고해야 합니다. 두 정책의 결과를 모두 저장하고, summary.csv는 선택한 정책을 보여줍니다. 일부 집단이 failed이면 파일을 모두 저장한 뒤 종료코드 2를 반환합니다.

개별 ATET의 산업별 점수에 집계 가중치를 곱해 **먼저 합산한 뒤** 표준오차를 구합니다. Large−SME 차이도 같은 산업 순서에서 기여도를 빼므로 공분산을 반영합니다. 정규근사 신뢰구간·점별 p값을 출력하며 multiplier bootstrap은 없습니다. **집계 비중과 포함 지점은 고정된 것으로 취급합니다.** 비중 추정·사후 기준점 선택·반복 간 변동의 추가 불확실성을 별도 반영하지 않습니다. 논문의 ATET^o에 대한 완전한 추론 재현으로 해석하지 마십시오.

## 결과

- `summary.csv`: 선택한 strict/supported 정책의 연도별 ATET^o, 표준오차, 신뢰구간, p값, 집계 coverage
- `estimates.csv`: 두 정책의 전체 결과
- `atet_by_dose.csv`: 기준점별·연도별·관측 처치강도별 ATET, ESS, trimming, 실패 이유
- `reference_*/연도/cluster_scores.npz`: 산업 순서와 집계 기여도
- `reference_*/연도/audit.json`: 제외 전후 산업·기업 수, 폴드별 실제 학습 산업, 대역폭, 캐시 정보
- `report.html`, `ATET_overall_reference_*.png`: 표와 연도별 그래프
- `config.json`: 선택한 모델의 파라미터까지 확정한 실행 설정. 입력 설정과 중복된 settings.json은 만들지 않습니다.

원자료 Excel은 `data/main_data`에 포함되어 있습니다. 새 패키지에는 Git 이력, 기존 결과·분석 스크립트·캐시, Python 가상환경을 복사하지 않았습니다.

검증 내역은 `VALIDATION.md`에 있습니다. 테스트 실행:

```powershell
..\ACR\Scripts\python.exe -B -m unittest discover -s tests -v
```
