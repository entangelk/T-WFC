<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# T-WFC

Tensor Wave Function Collapse(`T-WFC`)는 Wave Function Collapse의 `중첩 -> 관측 -> 붕괴 -> 전파` 루프를 초소형 신경망 학습에 옮겨와, 경사하강법 없이도 학습이 가능한지 검증하는 연구용 프로토타입입니다.

<p align="center">
  <img src="./docs/media/make_moons_clean.gif" alt="T-WFC clean collapse on make_moons" width="900">
</p>
<p align="center"><sub><code>make_moons</code>에서의 안정적인 부분 붕괴 예시입니다. 32개 weight 중 8개가 commit되었고 rollback 압력 없이 decision boundary가 단계적으로 바뀝니다.</sub></p>

아래에 들어가는 시각화는 전부 실제 CLI 실행으로 생성한 산출물이며, 설명용 목업 이미지가 아닙니다.

## 이 README에서 보여주려는 것

- 검색이 안정적일 때 붕괴 경로가 어떻게 보이는지
- 모순이 많이 발생할 때 rollback과 forced commit이 어떤 식으로 개입하는지
- 왜 지금의 시각화 스택이 단일 최종 결과 그림보다 더 많은 정보를 주는지

## 안정 경로 vs 압력 경로

<table>
  <tr>
    <td width="50%">
      <img src="./docs/media/make_moons_clean.gif" alt="안정적인 붕괴 GIF" width="100%">
    </td>
    <td width="50%">
      <img src="./docs/media/make_moons_stress.gif" alt="모순이 많은 복구 GIF" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>안정 경로</strong><br>
      직접 commit이 대부분을 차지합니다. rollback burst 없이 boundary가 또렷해집니다.
    </td>
    <td valign="top">
      <strong>모순이 많은 복구 경로</strong><br>
      같은 toy setup이라도 tolerance를 거칠게 주면 rollback pressure, alt-choice retry, forced commit이 눈에 띄게 등장합니다.
    </td>
  </tr>
</table>

이 프로젝트의 핵심은 “최종 classifier가 나왔는가”만이 아닙니다. discrete weight state가 붕괴되는 동안 검색이 어떤 행동을 보였는지가 같이 중요합니다.

## 예전 정적 시각화 vs 현재 시각화

<table>
  <tr>
    <td width="50%">
      <img src="./docs/media/make_moons_clean_overview.png" alt="정적 최종 상태 overview" width="100%">
    </td>
    <td width="50%">
      <img src="./docs/media/make_moons_clean_storyboard.png" alt="이벤트 인지형 storyboard" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>예전 정적 view</strong><br>
      initial shadow, final shadow, final hard를 비교하면서 <em>어디까지 갔는가</em>를 보여주기 좋습니다.
    </td>
    <td valign="top">
      <strong>현재 이벤트 인지형 view</strong><br>
      commit 기준 snapshot, event badge, ban overlay, search-pressure 정보까지 함께 보여주므로 <em>어떻게 거기까지 갔는가</em>를 읽기 좋습니다.
    </td>
  </tr>
</table>

## Stress Case: 왜 복구 로직이 중요한가

<table>
  <tr>
    <td width="50%">
      <img src="./docs/media/make_moons_stress_storyboard.png" alt="rollback과 forced commit이 보이는 stress storyboard" width="100%">
    </td>
    <td width="50%">
      <img src="./docs/media/make_moons_stress_metrics.png" alt="stress metrics timeline" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>Storyboard</strong><br>
      committed history 안에서 `ROLLBACK`, `ALT`, `FORCED`, ban focus, frontier pressure가 어디에서 나타났는지 보여줍니다.
    </td>
    <td valign="top">
      <strong>Metrics timeline</strong><br>
      모순이 많은 경로가 흔들리긴 해도 결국 유의미한 hard-state classifier로 회복되는 과정을 보여줍니다.
    </td>
  </tr>
</table>

역사적인 차이도 있습니다. frontier 기반 forced-commit fallback이 들어가기 전에는 이 stress 설정이 `0/32` committed weights에서 끝날 수 있었습니다. 지금 시각화는 그 차이를 바로 보이게 하려고 만든 부분도 큽니다.

## Multi-Seed 동작

<p align="center">
  <img src="./docs/media/make_moons_seed_gallery.png" alt="make_moons multi-seed gallery" width="950">
</p>
<p align="center"><sub><code>make_moons</code> seed sweep 예시입니다. 안정적인 seed, 상대적으로 약한 seed, search-pressure 요약을 한 화면에서 비교할 수 있습니다.</sub></p>

GIF가 “한 번의 run에서 무슨 일이 일어났는가?”를 보여준다면, gallery는 “seed만 바꿨을 때 얼마나 흔들리는가?”를 보여줍니다. 생성 예시 report는 [docs/media/make_moons_seed_report.md](./docs/media/make_moons_seed_report.md)에서 볼 수 있습니다.

## T-WFC vs SGD 비교

<p align="center">
  <img src="./docs/media/make_moons_twfc_vs_sgd.gif" alt="T-WFC vs SGD on make_moons" width="950">
</p>
<p align="center"><sub><code>make_moons</code>에서는 T-WFC 붕괴 진행과 고정된 SGD 최종 경계를 한 번에 보여줍니다. 과정과 baseline을 동시에 읽기 가장 좋은 비교입니다.</sub></p>

<table>
  <tr>
    <td width="50%">
      <img src="./docs/media/make_moons_twfc_vs_sgd_boundaries.png" alt="T-WFC vs SGD boundary board on make_moons" width="100%">
    </td>
    <td width="50%">
      <img src="./docs/media/spiral_twfc_vs_sgd.gif" alt="T-WFC vs SGD on spiral" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>발표용으로 가장 좋은 케이스: make_moons</strong><br>
      T-WFC hard test accuracy는 <code>0.950</code>, SGD는 <code>0.975</code>까지 갑니다. gap이 작고 collapse 과정도 읽기 쉬워서 가장 눈에 잘 들어옵니다.
    </td>
    <td valign="top">
      <strong>확장 한계를 보여주는 케이스: spiral</strong><br>
      symmetry-breaking jitter 덕분에 더 깊은 모델도 이제 실제로 움직이지만, GIF를 보면 아직 SGD와의 품질 차이가 분명합니다. T-WFC hard test accuracy는 <code>0.367</code>, SGD는 <code>0.422</code>입니다.
    </td>
  </tr>
</table>

<p align="center">
  <img src="./docs/media/iris_twfc_vs_sgd_metrics.png" alt="T-WFC vs SGD metrics on iris" width="920">
</p>
<p align="center"><sub><code>iris</code>는 metric gap을 가장 깔끔하게 보여줍니다. T-WFC shadow test accuracy는 <code>0.833</code>, hard test accuracy는 <code>0.639</code>, SGD는 <code>0.944</code>입니다.</sub></p>

누군가 “T-WFC가 흥미로운 일을 하긴 하나?”라고 물으면 `make_moons` GIF를 보여주면 됩니다.

누군가 “이미 SGD처럼 확장되나?”라고 물으면 `spiral`과 `iris`를 보여주면 됩니다.

## 현재 상태

- `make_moons`, `spiral`, vendored `iris.csv`를 지원합니다.
- 모델 경로는 기존 단일 hidden layer toy MLP뿐 아니라 `2-24-24-3`, `4-16-16-3` 같은 다층 구성도 지원합니다.
- 다층 run에서는 single-weight observation이 완전히 평평한 zero-signal 상태에 갇히지 않도록 작은 symmetry-breaking initial jitter가 기본 적용됩니다.
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
- CLI는 같은 MLP를 `numpy` SGD baseline으로도 학습시켜서 T-WFC와 전통적인 optimizer 경로를 바로 비교할 수 있습니다.
- 2D 데이터셋은 `T-WFC vs SGD` boundary board와 comparison GIF까지 저장할 수 있고, 모든 데이터셋은 `T-WFC vs SGD` metrics board를 저장할 수 있습니다.
- 패키지는 이제 `pyproject.toml`을 통해 설치형 `t-wfc` CLI를 제공합니다.

## 빠른 시작

```bash
python3 -m pip install -e .
t-wfc --help
PYTHONPATH=src python3 -m unittest discover -s tests
t-wfc --dataset make_moons --max-steps 8 --show-steps 6
t-wfc --dataset spiral --samples 240 --hidden-layers 24,24 --max-steps 18 --show-steps 6
t-wfc --dataset iris --hidden-layers 16,16 --max-steps 18 --compare-sgd --sgd-epochs 160 --sgd-batch-size 24 --show-steps 6
t-wfc --dataset make_moons --samples 160 --hidden-layers 12,12 --max-steps 12 --compare-sgd --sgd-epochs 140 --sgd-batch-size 24 --save-baseline-metrics-plot artifacts/make_moons/plots/twfc_vs_sgd_metrics.png --save-baseline-comparison-plot artifacts/make_moons/plots/twfc_vs_sgd_boundaries.png --save-baseline-comparison-gif artifacts/make_moons/animations/twfc_vs_sgd.gif --max-frame-count 6 --gif-frame-duration-ms 320
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-plot artifacts/make_moons/plots/overview.png --save-progress-plot artifacts/make_moons/plots/progress.png --progress-panels 5 --save-metrics-plot artifacts/make_moons/plots/metrics.png --save-frames-dir artifacts/make_moons/frames/steps --max-frame-count 6
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-storyboard artifacts/make_moons/plots/storyboard.png --storyboard-panels 5 --save-gif artifacts/make_moons/animations/steps.gif --max-frame-count 6 --gif-frame-duration-ms 350
t-wfc --dataset make_moons --max-steps 4 --backtrack-tolerance -10 --rollback-depth 1 --max-frontier-rollbacks 1 --max-attempt-multiplier 12 --show-steps 4 --save-metrics-plot artifacts/make_moons/plots/stress_metrics.png --save-storyboard artifacts/make_moons/plots/stress_storyboard.png --storyboard-panels 5 --save-gif artifacts/make_moons/animations/stress.gif --max-frame-count 5 --gif-frame-duration-ms 420
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
- `src/t_wfc/model.py`: 단일/다층 MLP 정의와 SGD baseline용 backprop 지원
- `src/t_wfc/baseline.py`: 비교용 `numpy` SGD baseline 학습
- `src/t_wfc/state.py`: 이산 확률 상태
- `src/t_wfc/trainer.py`: 붕괴 루프, rollback 로직, 메트릭, snapshot
- `src/t_wfc/batch.py`: seed list 기반 반복 실험 실행과 seed별 artifact export
- `src/t_wfc/reporting.py`: drill-down 링크가 포함된 Markdown seed-sweep report 생성
- `src/t_wfc/visualization.py`: overview, progress, metrics, storyboard, GIF, seed gallery, `T-WFC vs SGD` 비교 plot 생성
- `src/t_wfc/cli.py`: 커맨드라인 진입점
- `docs/media/`: 이 README에서 직접 사용하는 공개용 시각화 샘플 모음
- `pyproject.toml`: 패키지 메타데이터, 의존성, `t-wfc` console script

## 참고

- 아직 완성된 프레임워크가 아니라 연구용 프로토타입입니다.
- 런타임 핵심 의존성은 `numpy`입니다.
- 시각화 출력에는 `matplotlib`를 사용합니다.
- GIF export에는 `Pillow`를 사용합니다.
