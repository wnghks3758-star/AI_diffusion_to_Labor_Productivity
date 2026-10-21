# 추가 무작위 탐색: 후보 201~400

사용자의 추가 200개 실행 요청에 따라 기존 난수 흐름을 이어 후보 201~400을 새 결과 계산 전에 고정했다. 기존 1~200의 설정·결과는 보존한다. 새 후보는 앞선 200개와 설정 식별자가 중복되지 않으며, 기존 생성기 seed 20261021에서 처음 200개를 정확히 재현한 뒤 다음 200개를 사용한다. 수치 계산 소스와 데이터 해시도 동일하다.

저장 필터, 자동 종료 기준, 모형·변수 및 지지 설정 범위는 `RANDOM_SEARCH_KO.md`와 동일하다. 사전 효과 8개가 모두 비유의하고 2023년 전체 효과가 유의한 음수인 경우만 상세 보관한다. 전체 목표를 만족하는 첫 사례가 나오면 즉시 멈추고, 그렇지 않으면 추가 200개에서 종료한다. 추정 오류도 시도에 포함하며 다른 후보로 대체하지 않는다.

## 실행 및 재개

```bash
# 최초 한 번: 기존 흐름을 재현하고 다음 200개만 별도 계획으로 고정
.venv/bin/python -m dml_acr.random_experiments continue-plan --previous results/random200 --output results/random201_400 --max-attempts 200

# 실행 또는 같은 계획 재개
.venv/bin/python -m dml_acr.random_experiments search --output results/random201_400

# 종료 후 저장 조건·수치·시도 수·자동 종료 감사
.venv/bin/python scripts/validate_random_results.py --root results/random201_400

# 기존 200개와 새 배치를 결합하고 최신 누적 보정 재계산
.venv/bin/python -m dml_acr.random_history --root results/random201_400

# 새 배치의 한국어 Word/PDF 보고서
python scripts/build_random_report.py --root results/random201_400 --output work/random201_400_reporting/output
```

배치 내부 `attempt_number`는 1부터 시작하지만 후보 ID는 `random_201`부터다. 누적 표의 `global_attempt_number`는 201부터 시작한다. `plan.json`에 이전 계획·장부의 해시, 생성기 offset 200, 기존 설정 수 264 및 이전 배치 경로를 남겼다. 기존 계획이나 원자료를 바꿔서 새 후보를 생성하지 않는다.

## 다중 탐색 보정과 누적 파일

이번 배치의 200×20=4,000개 검정 Bonferroni와, 이전 264개 설정 및 이번 200개 설정의 464×20=9,280개 검정 Bonferroni를 제공한다. 이전 264개는 최초 모형·변수 탐색 64개와 첫 무작위 후보 200개다. 추정 불가·탈락·후속 연도 미계산도 계획 분모에 포함한다.

`results/random201_400/cumulative/`에는 다음을 기록한다.

- `all_random_attempts.csv`: 두 배치의 모든 시도·통과·탈락·오류와 전역 시도 번호
- `all_retained_estimates.csv`: 두 배치의 저장 조건 통과 사례만 결합한 상세 결과
- `cumulative_summary.json`: 누적 시도·통과 수, 중복 여부, 원본 계획의 해시와 보정 분모

누적 추정 표의 `p_bonferroni_all_random_budget`는 무작위 400개만 포함한 8,000개 검정 보정이다. `p_bonferroni_cumulative_budget`는 64개 선행 탐색까지 포함한 9,280개 보정으로 두 배치 모두 다시 계산한다. `p_bonferroni_cumulative_at_batch`에는 원본 배치 시점에 적용했던 보정 값을 별도로 남긴다. 첫 배치의 원본 결과 파일과 보고서는 수정하지 않는다.

새 배치의 상세 결과는 `accepted/`에만 보관하고 탈락 상세 파일은 제거한다. Google Drive `Labor_Productivity/Random_Search_201_400`에 계획·중간 체크포인트·최종 결과와 보고서를 올린다. 통과 사례는 결과를 보고 선택된 탐색 사례이며 사전 비유의는 평행추세의 입증이 아니다.
