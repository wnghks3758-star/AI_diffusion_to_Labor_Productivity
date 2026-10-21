# 무작위 탐색 추가 결과: 후보 401~600

이번 배치 200/200개를 시도했습니다. 상세 저장 7개, 조건 탈락 131개, 실행 오류 62개입니다.
종료 사유: `budget_exhausted`. 전체 목표 충족 0개, 우수 조건 포함 0개입니다.

사전 2020·2021년의 전체/SME/Large/조절효과 8개가 모두 p≥.05이고, 2023년 전체 효과가 음수이며 p<.05인 경우만 상세 보관했습니다. 이후 회복·SME의 조기 양수 전환·음수 조절효과 경로까지 만족해야 전체 목표입니다. SME 2025년 유의한 양수는 우수 조건입니다.

## 전체 배치 누적

누적 무작위 시도 600개, 상세 보관 25개, 조건 탈락 380개, 실행 오류 195개입니다. 누적 전체 목표 충족 0개입니다.
계획 후보 600개의 설정 식별자가 모두 다릅니다. 원래 난수 흐름을 이어갔으며 이전 배치의 설정·결과를 보존했습니다.

## 이번 배치의 상세 보관 사례

| 후보 | 2023 전체 추정치 (p) | 2025 SME 추정치 (p) | 2025 조절효과 추정치 (p) | 전체 목표 |
|---|---:|---:|---:|---|
| random_409 | -0.2043 (0.0437) | -0.1875 (0.3748) | 0.0145 (0.9605) | False |
| random_508 | -0.1416 (0.0268) | -0.2759 (0.1025) | 0.0284 (0.8877) | False |
| random_521 | -0.1491 (0.0217) | -0.1927 (0.1326) | -0.0007 (0.9966) | False |
| random_540 | -0.1732 (0.0237) | -0.0723 (0.5425) | -0.1022 (0.5503) | False |
| random_559 | -0.1829 (0.0163) | -0.2655 (0.1269) | 0.0975 (0.6079) | False |
| random_572 | -0.1627 (0.0328) | -0.0678 (0.5699) | -0.0793 (0.6705) | False |
| random_577 | -0.1918 (0.0435) | -0.3486 (0.1358) | 0.1666 (0.5947) | False |

표는 명목 양측 p값입니다. 판정은 반올림 전 값으로 수행했습니다.
이번 배치 4,000개 검정 및 이전 464개 설정을 포함한 13,280개 검정의 Bonferroni 보정을 CSV와 보고서에 함께 수록했습니다.

## 저장 조건 통과 사례 안의 후속 조건별 충족 수

- recovery: 0개
- sme_crosses_positive_earlier: 0개
- sme_2025_significantly_positive: 0개
- moderation_negative_significant: 0개

탈락 사례의 후속 연도는 계산을 생략하므로 위 수는 전체 후보의 후속 효과 빈도가 아닙니다.

## 오류와 사용 자원

- Each industry stratum must contain >=2 industries; change dummy rules: 62개

Worker 실행 누적 579.5초, 최대 RSS 252.8 MiB입니다. 프로세스 시작·보고서 작성·업로드 등의 전체 대기 시간과 구분합니다.

## 상세 기록

- [실제 종료 상태](../results/random401_600/summary.json)
- [모든 후보 설정](../results/random401_600/plan.json)
- [시도 장부](../results/random401_600/all_attempts.csv)
- [통과 사례 추정치](../results/random401_600/accepted_estimates.csv)
- [누적 요약](../results/random401_600/cumulative/cumulative_summary.json)
- [누적 통과 사례 추정치](../results/random401_600/cumulative/all_retained_estimates.csv)
- [Drive 결과·보고서](https://drive.google.com/drive/folders/14iEdHHlM6I4dTCsRS3LEyAsyI7sFrQvD)

탐색 설계·재시작 및 보정은 RANDOM_CONTINUATION_KO.md를 참조하세요. 결과를 보고 선택한 탐색 사례이며 사전 비유의는 평행추세의 입증이 아닙니다. 조건부 부트스트랩 및 식별 제한은 METHODS_KO.md와 RANDOM_SEARCH_KO.md에 설명했습니다.
