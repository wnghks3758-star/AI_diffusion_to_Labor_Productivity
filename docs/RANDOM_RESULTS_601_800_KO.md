# 무작위 탐색 추가 결과: 후보 601~800

이번 배치 200/200개를 시도했습니다. 상세 저장 8개, 조건 탈락 119개, 실행 오류 73개입니다.
종료 사유: `budget_exhausted`. 전체 목표 충족 0개, 우수 조건 포함 0개입니다.

사전 2020·2021년의 전체/SME/Large/조절효과 8개가 모두 p≥.05이고, 2023년 전체 효과가 음수이며 p<.05인 경우만 상세 보관했습니다. 이후 회복·SME의 조기 양수 전환·음수 조절효과 경로까지 만족해야 전체 목표입니다. SME 2025년 유의한 양수는 우수 조건입니다.

## 전체 배치 누적

누적 무작위 시도 800개, 상세 보관 33개, 조건 탈락 499개, 실행 오류 268개입니다. 누적 전체 목표 충족 0개입니다.
계획 후보 800개의 설정 식별자가 모두 다릅니다. 원래 난수 흐름을 이어갔으며 이전 배치의 설정·결과를 보존했습니다.

## 이번 배치의 상세 보관 사례

| 후보 | 2023 전체 추정치 (p) | 2025 SME 추정치 (p) | 2025 조절효과 추정치 (p) | 전체 목표 |
|---|---:|---:|---:|---|
| random_614 | -0.2808 (0.0378) | -0.2023 (0.4238) | 0.0454 (0.8935) | False |
| random_629 | -0.2448 (0.0425) | -0.2030 (0.3456) | 0.0543 (0.8404) | False |
| random_632 | -0.3222 (0.0004) | -0.1689 (0.2390) | -0.1098 (0.5363) | False |
| random_660 | -0.2978 (0.0154) | 0.3400 (0.3139) | -0.8748 (0.0310) | False |
| random_688 | -0.2566 (0.0182) | -0.3242 (0.1614) | -0.0037 (0.9871) | False |
| random_720 | -0.1936 (0.0359) | -0.0688 (0.6376) | -0.0990 (0.6314) | False |
| random_765 | -0.3936 (0.0245) | -0.3473 (0.1965) | -0.0779 (0.8014) | False |
| random_796 | -0.1812 (0.0035) | -0.2090 (0.1185) | -0.0768 (0.6474) | False |

표는 명목 양측 p값입니다. 판정은 반올림 전 값으로 수행했습니다.
이번 배치 4,000개 검정 및 이전 664개 설정을 포함한 17,280개 검정의 Bonferroni 보정을 CSV와 보고서에 함께 수록했습니다.

## 저장 조건 통과 사례 안의 후속 조건별 충족 수

- recovery: 2개
- sme_crosses_positive_earlier: 1개
- sme_2025_significantly_positive: 0개
- moderation_negative_significant: 1개

탈락 사례의 후속 연도는 계산을 생략하므로 위 수는 전체 후보의 후속 효과 빈도가 아닙니다.

## 오류와 사용 자원

- Each industry stratum must contain >=2 industries; change dummy rules: 73개

Worker 실행 누적 477.9초, 최대 RSS 242.2 MiB입니다. 프로세스 시작·보고서 작성·업로드 등의 전체 대기 시간과 구분합니다.

## 상세 기록

- [실제 종료 상태](../results/random601_800/summary.json)
- [모든 후보 설정](../results/random601_800/plan.json)
- [시도 장부](../results/random601_800/all_attempts.csv)
- [통과 사례 추정치](../results/random601_800/accepted_estimates.csv)
- [누적 요약](../results/random601_800/cumulative/cumulative_summary.json)
- [누적 통과 사례 추정치](../results/random601_800/cumulative/all_retained_estimates.csv)
- [Drive 결과·보고서](https://drive.google.com/drive/folders/1ufW_SBD1Mytx6gPjDJ6Oo4FCdCO0SRAq)

탐색 설계·재시작 및 보정은 RANDOM_CONTINUATION_KO.md를 참조하세요. 결과를 보고 선택한 탐색 사례이며 사전 비유의는 평행추세의 입증이 아닙니다. 조건부 부트스트랩 및 식별 제한은 METHODS_KO.md와 RANDOM_SEARCH_KO.md에 설명했습니다.
