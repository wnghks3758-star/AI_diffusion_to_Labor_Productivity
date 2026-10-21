# 추가 무작위 탐색: 후보 601~800

사용자의 계속 실행 요청을 기존과 같은 200개 단위로 진행한다. 생성기 seed 20261021의 흐름을 이어 601~800번째 후보를 고정했다. 앞선 600개와 중복되지 않으며 저장·정지 조건과 모형·변수·지지 범위를 유지한다.

사전 2020·2021년 전체/SME/Large/조절효과 8개 p≥.05와 2023년 전체 효과의 유의한 음수(p<.05)를 만족한 경우만 상세 보관한다. 이후 회복, SME의 더 빠른 양수 전환, 음수 조절효과 경로까지 모두 만족하면 즉시 멈춘다. 2025년 SME의 유의한 양수는 우수 조건이다. 전체 목표를 못 찾으면 추가 200개에서 종료하며 추정 오류도 시도에 포함한다.

```bash
.venv/bin/python -m dml_acr.random_experiments continue-plan --previous results/random401_600 --output results/random601_800 --max-attempts 200
.venv/bin/python -m dml_acr.random_experiments search --output results/random601_800
.venv/bin/python scripts/validate_random_results.py --root results/random601_800
.venv/bin/python -m dml_acr.random_history --root results/random601_800
.venv/bin/python scripts/validate_random_history.py --root results/random601_800
python scripts/build_random_report.py --root results/random601_800 --output work/random601_800_reporting/output
```

이번 배치 보정은 4,000개 검정이다. 누적 무작위 계획 800개는 16,000개 검정이며, 최초 모형·변수 탐색 64개까지 포함한 누적 보정은 864×20=17,280개 검정이다. 추정 불가·탈락·후속 연도 미계산도 분모에 포함한다. 이전 배치 원본은 보존하고 누적 표에 최신 보정 값을 함께 제공한다.

계획·장부의 해시 연결, 800개 후보의 설정 식별자 중복, 실제 시도 수·전역 번호, 통과 사례의 수치·저장 조건을 감사한다. Google Drive `Labor_Productivity/Random_Search_601_800`에 계획, 중간 체크포인트, 최종 결과와 Word/PDF 보고서를 저장한다. 해석 및 상세 범위는 `RANDOM_SEARCH_KO.md`, `RANDOM_CONTINUATION_KO.md`와 `METHODS_KO.md`를 참조한다. 이 결과는 선택된 탐색 결과이며 사전 비유의는 평행추세의 입증이 아니다.
