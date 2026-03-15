<p align="center">
  <a href="./VERIFICATION.en.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./VERIFICATION.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# 검증 가이드

이 문서는 지금 있는 검사와 테스트가 왜 필요한지, 각각 무엇을 노리는지, 그리고 생성되는 리포트와 아티팩트를 앞으로 어떻게 정리하면 좋은지를 설명합니다.

## 왜 이런 방식으로 테스트하나

`T-WFC`는 일반적인 경사하강 기반 학습기가 아니라, 붕괴 루프 자체를 검증하는 연구용 프로토타입입니다. 그래서 테스트 우선순위도 조금 다릅니다.

- 성능 미세 조정보다 collapse loop 의미가 깨지지 않는지가 더 중요합니다.
- rollback, forced commit, snapshot bookkeeping 같은 조용한 회귀를 빨리 잡아야 합니다.
- 이 프로젝트는 시각화와 Markdown report도 산출물이라서 파일 출력 경로 자체도 검증해야 합니다.
- 알고리즘이 아직 빠르게 바뀌는 단계라 테스트는 작고 자주 돌릴 수 있어야 합니다.

## 현재 검사들이 노리는 것

### 1. 데이터셋 무결성

`make_moons`, Iris 로딩이 기대한 shape, label, split 동작을 내는지 확인합니다.

이 테스트가 막는 것:
- 데이터 생성 깨짐
- label coverage 누락
- train/test split 전제의 우발적 변경

### 2. Trainer 스모크 테스트

짧은 end-to-end 실행을 돌리고 다음을 확인합니다.

- loss가 유한한지
- accuracy가 정상 범위인지
- step 수가 기대와 맞는지
- snapshot과 최종 counter가 맞물리는지

이 테스트가 막는 것:
- collapse loop 회귀
- metric 계산 버그
- result/snapshot 요약 구조 깨짐

### 3. Rollback / 모순 스트레스 테스트

음수 tolerance 같은 가혹한 설정으로 rollback이 많이 일어나게 해서, 평소에는 잘 안 타는 경로를 일부러 태웁니다.

이 테스트가 막는 것:
- rollback 경로 회귀
- forced commit fallback 실패
- forbidden-value ban 추적 오류
- frontier pressure 집계 오류

왜 중요한가:
이 프로젝트의 어려운 버그는 깔끔한 run보다 탐색이 막히거나 모순나는 상황에서 더 자주 나옵니다.

### 4. 시각화 출력 테스트

PNG, GIF를 실제로 쓰고 파일이 생성되며 비어 있지 않은지만 확인합니다.

이 테스트가 막는 것:
- plotting 회귀
- export wiring 실패
- serialization 오류

하지만 아직 보장하지 않는 것:
- 미적 완성도
- 픽셀 단위 정답
- 발표용으로 충분히 읽기 좋은지 여부

그 부분은 사람 눈으로 봐야 합니다.

### 5. 배치 리포트 테스트

다음을 확인합니다.

- seed gallery 생성
- seed별 artifact export
- Markdown report 생성
- best/worst highlight
- peak-ban, latest-ban 요약

이 테스트가 막는 것:
- 연구용 reporting workflow 붕괴
- gallery/report/seed artifact 간 drill-down 링크 깨짐
- multi-seed 요약 로직 오류

## 어떻게 검증하나

### 기본 스모크 테스트

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

### 문법/임포트 sanity check

```bash
python3 -m py_compile src/t_wfc/*.py tests/test_smoke.py
```

### 패키징 / CLI sanity check

설치형 CLI 경로까지 확인하고 싶다면:

```bash
python3 -m pip install -e .
t-wfc --help
```

이 검사는 `pyproject.toml` 메타데이터와 console entry point가 제대로 연결돼 있는지 확인해줍니다.

## 현재 테스트가 아직 보장하지 않는 것

- 알고리즘이 수치적으로 최적이라는 보장
- 모든 seed에서 정확한 accuracy 목표 고정
- 이미지 golden reference 비교
- runtime/memory benchmark
- storyboard, GIF, report가 실제로 읽기 좋은지에 대한 사람 수준 평가

이건 의도적인 한계입니다. 아직은 개념 검증 단계이기 때문입니다.

## 리포트가 계속 늘어날 때의 정리 기준

데이터셋과 seed sweep이 늘어나면 리포트도 계속 커집니다. 그래서 생성물은 dataset 기준으로 묶는 편이 좋습니다.

권장 구조:

```text
artifacts/
  make_moons/
    plots/
    animations/
    frames/
    reports/
  iris/
    plots/
    reports/
```

권장 원칙:
- 생성 산출물은 `artifacts/<dataset>/...` 아래에 둡니다.
- 영구 문서는 `docs/` 아래에 둡니다.
- reference 문서와 generated run output을 섞지 않습니다.

## 퍼블릭 배포에서 왜 이 문서가 중요한가

퍼블릭 저장소에서는 새로 들어온 사람이 빠르게 세 가지를 확인할 수 있어야 합니다.

1. 이 코드를 실제로 실행할 수 있는가?
2. 기본 연구 루프가 아직 멀쩡하게 돌아가는가?
3. 시각화와 리포트가 무엇을 보여주는지 이해할 수 있는가?

이 문서는 그 답을 테스트 코드를 뒤집어보지 않고도 파악할 수 있게 해주기 위해 존재합니다.
