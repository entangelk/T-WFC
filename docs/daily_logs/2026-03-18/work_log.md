# Work Log - 2026-03-18

## Goal

- 최종 실험: 선형/비선형 데이터셋을 확장하여 T-WFC의 적용 범위를 명확히 규정
- 속도/메모리 벤치마크로 T-WFC의 실용적 이점 여부 확인
- 실험 결과 문서(RESULT.md) 작성으로 프로젝트 마무리

## Completed Work

- `CLAUDE.md`, `AGENTS.md`에 섹션 6.7 "흔들리지 않는 측정 설계" 추가
  - 결정론적 재현성, 외부 상태 격리, 데이터 경로 일관성, 테스트 보장 원칙
  - 기존 6.7 → 6.8로 번호 이동
- 새 데이터셋 5종 추가 (`src/t_wfc/data.py`)
  - `linear_binary`: 대각선 분리, 선형 2클래스
  - `blobs_binary`: 가우시안 2클러스터, 선형 2클래스
  - `make_blobs`: 가우시안 3클러스터, 선형 3클래스
  - `xor`: XOR 패턴, 비선형 2클래스
  - `circles`: 동심원, 비선형 2클래스
- CLI에 새 데이터셋 연결 (`src/t_wfc/cli.py`)
- 8개 데이터셋 전체 비교 실험 수행 (seed=7, SGD epochs=300)
  - linear_binary: T-WFC 0.967 = SGD 0.967
  - blobs_binary: T-WFC 1.000 = SGD 1.000
  - make_blobs: T-WFC 1.000 = SGD 1.000 (5 seed 모두 동일)
  - iris: T-WFC 0.972 > SGD 0.944
  - make_moons: T-WFC 0.933 < SGD 1.000
  - xor: T-WFC 0.660 < SGD 1.000
  - circles: T-WFC 0.620 < SGD 1.000
  - spiral: T-WFC 0.433 < SGD 0.987
- 속도/메모리 벤치마크 측정
  - 소형 모델(32~67 params): 실행 시간 비슷, T-WFC가 때로 빠름
  - 대형 모델(747 params, spiral): T-WFC 16.7× 느림, 117× 메모리
  - 결론: 속도 우위 없음, 메모리는 항상 열세
- `docs/RESULT.md` (한국어), `docs/RESULT.en.md` (영어) 작성
  - 정확도 비교표, 속도/메모리 비교표, 결론, 한계, 가능성, 공정성 선언
- 전 2D 데이터셋 비교 GIF 생성 (7개: linear_binary, blobs_binary, make_blobs, make_moons, xor, circles, spiral)
- `README.md`, `README.ko.md` 최종 정리
  - "T-WFC vs SGD+Momentum" → "Final Results: T-WFC vs SGD+Momentum"으로 개편
  - 8개 데이터셋 정확도+속도 비교표 추가
  - 선형 성공(3 GIF) / 비선형 실패(4 GIF) 시각화 섹션으로 재구성
  - 결론 단락 추가, RESULT.md 링크

## Technical Decisions

- make_blobs를 선형 분리 가능 대표 데이터셋으로 선택: 클러스터 중심을 원 위에 등간격 배치, std=0.6으로 겹침 없이 분리 가능
- xor/circles를 비선형 대표로 선택: 각각 사분면 패턴과 동심원 — 단순하지만 선형 분류기로는 불가능한 구조
- 벤치마크에서 T-WFC forward 횟수는 상한 추정 (n_weights × obs_budget × domain_size)

## Verification

- `python3 -m py_compile src/t_wfc/data.py src/t_wfc/cli.py` — 통과
- `PYTHONPATH=src python3 -m unittest discover -s tests` — 36 tests passed
- make_blobs 5 seed (3, 7, 13, 21, 42) 모두 T-WFC 1.000 확인
- 모든 비교 실험 seed=7 고정, 동일 DatasetSplit 객체 공유 확인

## Next Steps

- 프로젝트 마무리 단계. 추가 실험 방향이 필요하면 RESULT.md "가능성이 남은 방향" 참고.
