<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# T-WFC

Tensor Wave Function Collapse(`T-WFC`)는 Wave Function Collapse의 `중첩 -> 관측 -> 붕괴 -> 전파` 루프를 초소형 신경망 학습에 옮겨와, 경사하강법 없이도 학습이 가능한지 검증하는 연구용 프로토타입입니다.

## 현재 상태

- `make_moons`와 vendored `iris.csv`를 모두 지원합니다.
- trainer에는 관측, 단일 weight 붕괴, 전파, rollback-aware backtracking, hybrid score 기반 forced commit이 들어가 있습니다.
- `make_moons` 실행은 다음 시각화를 저장할 수 있습니다.
  - `initial shadow / final shadow / final hard` 3패널 overview plot
  - committed collapse step 기준 progress timeline plot
  - step-by-step 확인용 per-snapshot frame sequence
  - metrics, selected snapshot, event highlight를 함께 보여주는 storyboard
  - event badge와 누적 카운터가 들어간 committed snapshot 기반 GIF animation
  - rollback burst, alt-choice retry, forced commit이 한 태그가 아니라 분리된 신호로 표시됩니다
  - forbidden-value ban과 frontier pressure도 별도 overlay 신호로 표시됩니다
  - ban overlay는 이제 ban 개수뿐 아니라 어떤 weight에 ban이 쌓였는지도 함께 보여줍니다
- 어떤 데이터셋이든 shadow/hard loss, accuracy를 보는 metrics timeline plot을 저장할 수 있습니다.
- multi-seed 실행은 comparison gallery, seed별 drill-down artifact, 그리고 best/worst seed highlight, peak-ban summary, metrics/storyboard/GIF 링크가 들어간 Markdown report를 저장할 수 있습니다.
- 패키지는 이제 `pyproject.toml`을 통해 설치형 `t-wfc` CLI를 제공합니다.

## 빠른 시작

```bash
python3 -m pip install -e .
t-wfc --help
PYTHONPATH=src python3 -m unittest discover -s tests
t-wfc --dataset make_moons --max-steps 8 --show-steps 6
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-plot artifacts/make_moons/plots/overview.png --save-progress-plot artifacts/make_moons/plots/progress.png --progress-panels 5 --save-metrics-plot artifacts/make_moons/plots/metrics.png --save-frames-dir artifacts/make_moons/frames/steps --max-frame-count 6
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-storyboard artifacts/make_moons/plots/storyboard.png --storyboard-panels 5 --save-gif artifacts/make_moons/animations/steps.gif --max-frame-count 6 --gif-frame-duration-ms 350
t-wfc --dataset make_moons --max-steps 8 --seed-list 7,11,17,23,31 --save-seed-gallery artifacts/make_moons/plots/seed_gallery.png --gallery-columns 3 --save-seed-artifacts-dir artifacts/make_moons/reports/seed_runs --save-md-report artifacts/make_moons/reports/seed_report.md --report-title "T-WFC make_moons Seed Sweep"
```

## 문서

- 컨셉 문서, 영어: [docs/CONCEPT.en.md](./docs/CONCEPT.en.md)
- 컨셉 문서, 한국어: [docs/CONCEPT.md](./docs/CONCEPT.md)
- 검증 가이드, 영어: [docs/VERIFICATION.en.md](./docs/VERIFICATION.en.md)
- 검증 가이드, 한국어: [docs/VERIFICATION.md](./docs/VERIFICATION.md)
- 변경 이력: [CHANGELOG.md](./CHANGELOG.md)

## 저장소 구조

- `src/t_wfc/data.py`: 데이터셋 로딩과 split
- `src/t_wfc/model.py`: toy MLP 정의
- `src/t_wfc/state.py`: 이산 확률 상태
- `src/t_wfc/trainer.py`: 붕괴 루프, rollback 로직, 메트릭, snapshot
- `src/t_wfc/batch.py`: seed list 기반 반복 실험 실행과 seed별 artifact export
- `src/t_wfc/reporting.py`: drill-down 링크가 포함된 Markdown seed-sweep report 생성
- `src/t_wfc/visualization.py`: overview, progress, metrics, storyboard, GIF, frame sequence, seed gallery plot
- `src/t_wfc/cli.py`: 커맨드라인 진입점
- `pyproject.toml`: 패키지 메타데이터, 의존성, `t-wfc` console script

## 참고

- 아직 완성된 프레임워크가 아니라 연구용 프로토타입입니다.
- 런타임 핵심 의존성은 `numpy`입니다.
- 시각화 출력에는 `matplotlib`를 사용합니다.
- GIF export에는 `Pillow`를 사용합니다.
