# 무작위 탐색 추가 결과: 후보 201~400

이번 배치 200/200개를 시도했습니다. 상세 저장 8개, 조건 탈락 125개, 실행 오류 67개입니다.
종료 사유: `budget_exhausted`. 전체 목표 충족 0개, 우수 조건 포함 0개입니다.

사전 2020·2021년의 전체/SME/Large/조절효과 8개가 모두 p≥.05이고, 2023년 전체 효과가 음수이며 p<.05인 경우만 상세 보관했습니다. 이후 회복·SME의 조기 양수 전환·음수 조절효과 경로까지 만족해야 전체 목표입니다. SME 2025년 유의한 양수는 우수 조건입니다.

## 두 배치 누적

누적 무작위 시도 400개, 상세 보관 18개, 조건 탈락 249개, 실행 오류 133개입니다. 누적 전체 목표 충족 0개입니다.
두 배치 계획 후보 400개의 설정 식별자가 모두 다릅니다. 원래 난수 흐름을 이어갔으며 첫 배치의 설정·결과를 보존했습니다.

## 이번 배치의 상세 보관 사례

| 후보 | 2023 전체 추정치 (p) | 2025 SME 추정치 (p) | 2025 조절효과 추정치 (p) | 전체 목표 |
|---|---:|---:|---:|---|
| random_212 | -0.1915 (0.0373) | -0.1659 (0.4415) | -0.0386 (0.8862) | False |
| random_234 | -0.3452 (0.0081) | -0.1844 (0.4107) | -0.1741 (0.5177) | False |
| random_265 | -0.3232 (0.0380) | -0.4857 (0.2169) | 0.0905 (0.8338) | False |
| random_268 | -0.1236 (0.0493) | -0.2924 (0.0491) | 0.0632 (0.7383) | False |
| random_315 | -0.5678 (0.0055) | -0.2609 (0.2789) | -0.1782 (0.4895) | False |
| random_320 | -0.2593 (0.0199) | -0.2388 (0.2485) | -0.2258 (0.5169) | False |
| random_330 | -0.1898 (0.0290) | -0.1387 (0.4369) | -0.0347 (0.8822) | False |
| random_342 | -0.4239 (0.0112) | 0.4547 (0.2158) | -0.6949 (0.0668) | False |

표는 명목 양측 p값입니다. 판정은 반올림 전 값으로 수행했습니다.
이번 배치 4,000개 검정 및 이전 264개 설정을 포함한 9,280개 검정의 Bonferroni 보정을 CSV와 보고서에 함께 수록했습니다.

## 저장 조건 통과 사례 안의 후속 조건별 충족 수

- recovery: 2개
- sme_crosses_positive_earlier: 2개
- sme_2025_significantly_positive: 0개
- moderation_negative_significant: 0개

탈락 사례의 후속 연도는 계산을 생략하므로 위 수는 전체 후보의 후속 효과 빈도가 아닙니다.

## 오류와 사용 자원

- Each industry stratum must contain >=2 industries; change dummy rules: 67개

Worker 실행 누적 475.3초, 최대 RSS 251.4 MiB입니다. 프로세스 시작·보고서 작성·업로드 등의 전체 대기 시간과 구분합니다.

## 상세 기록

- [실제 종료 상태](../results/random201_400/summary.json)
- [모든 후보 설정](../results/random201_400/plan.json)
- [시도 장부](../results/random201_400/all_attempts.csv)
- [통과 사례 추정치](../results/random201_400/accepted_estimates.csv)
- [누적 요약](../results/random201_400/cumulative/cumulative_summary.json)
- [누적 통과 사례 추정치](../results/random201_400/cumulative/all_retained_estimates.csv)
- [Drive 결과·보고서](https://drive.google.com/drive/folders/1XvDSYuUBfhrkLb0t1F5dwsXW46N3nt1s)

탐색 설계·재시작 및 보정은 RANDOM_CONTINUATION_KO.md를 참조하세요. 결과를 보고 선택한 탐색 사례이며 사전 비유의는 평행추세의 입증이 아닙니다. 조건부 부트스트랩 및 식별 제한은 METHODS_KO.md와 RANDOM_SEARCH_KO.md에 설명했습니다.
