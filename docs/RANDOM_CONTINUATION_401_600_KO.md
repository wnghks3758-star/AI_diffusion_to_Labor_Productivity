# 추가 무작위 탐색: 후보 401~600

사용자가 요청한 추가 200개를 기존 생성기 흐름(seed 20261021)의 401~600번째 후보로 고정했다. 앞선 400개와 중복되지 않으며, 동일한 변수·설정 범위와 저장·정지 조건을 사용한다. 결과변수, 처치, 기준 연도, 유의수준, 규모 분류, bootstrap·모형 seed도 유지한다.

상세 저장 조건은 2020·2021년 전체/SME/Large/조절효과 8개의 비유의(p≥.05)와 2023년 전체 효과의 유의한 음수(p<.05)다. 이후 단조 회복, SME의 더 빠른 양수 전환, 음수 조절효과 경로까지 만족하면 즉시 정지한다. SME 2025년 유의한 양수는 우수 조건이다. 전체 목표를 찾지 못하면 추가 200개에서 종료한다. 추정 오류도 시도에 포함한다.

```bash
.venv/bin/python -m dml_acr.random_experiments continue-plan --previous results/random201_400 --output results/random401_600 --max-attempts 200
.venv/bin/python -m dml_acr.random_experiments search --output results/random401_600
.venv/bin/python scripts/validate_random_results.py --root results/random401_600
.venv/bin/python -m dml_acr.random_history --root results/random401_600
.venv/bin/python scripts/validate_random_history.py --root results/random401_600
python scripts/build_random_report.py --root results/random401_600 --output work/random401_600_reporting/output
```

원자료와 수치 계산 소스 해시는 기존과 같다. 세 배치의 계획·장부를 해시 연결로 감사하고, 후보 식별자 중복과 전역 시도 번호를 검사한다. 이전 두 배치의 원본은 보존한다. 실행·변수 범위 및 해석은 `RANDOM_SEARCH_KO.md`와 `RANDOM_CONTINUATION_KO.md`를 따른다.

이번 배치는 4,000개 검정 보정을, 누적 표는 무작위 600개에 대한 12,000개 검정 보정을 제공한다. 최초 모형·변수 탐색 64개까지 포함한 누적 보정은 664×20=13,280개 검정이다. 실행 불가·탈락·후속 연도 미계산도 계획 분모에 포함한다. 이전 배치의 보정 값은 누적 표에서 별도 열에 보관하고 최신 보정을 공통 적용한다.

Google Drive `Labor_Productivity/Random_Search_401_600`에 계획, 중간 체크포인트, 최종 결과와 Word/PDF 보고서를 저장한다. 상세 결과는 저장 조건을 통과한 사례만 남기며 전체 시도의 설정·탈락 사유는 감사 장부에 기록한다. 결과를 보고 선택한 탐색이며 사전 비유의는 평행추세의 입증이 아니다.
