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

아래 시각화는 전부 실제 CLI 실행으로 생성하여 `docs/media/`에 커밋된 산출물이며, 설명용 목업이 아닙니다.

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
      같은 toy setup에 tolerance를 거칠게 설정하면 rollback pressure, alt-choice retry, forced commit이 눈에 띄게 등장합니다.
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
      initial shadow, final shadow, final hard 상태를 비교합니다 — <em>run이 어디서 끝났는지</em>를 보여줍니다.
    </td>
    <td valign="top">
      <strong>현재 이벤트 인지형 view</strong><br>
      commit 기준 snapshot에 event badge, ban overlay, search-pressure 정보가 붙어 있어서 <em>어떻게 거기까지 갔는지</em>를 보여줍니다.
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

Frontier 기반 forced-commit fallback이 들어가기 전에는 이 stress 설정이 `0/32` committed weights에서 끝날 수 있었습니다. 위 시각화를 보면 그 차이가 바로 드러납니다.

## Multi-Seed 동작

<p align="center">
  <img src="./docs/media/make_moons_seed_gallery.png" alt="make_moons multi-seed gallery" width="950">
</p>
<p align="center"><sub><code>make_moons</code> seed sweep 예시입니다. 안정적인 seed, 상대적으로 약한 seed, search-pressure 요약을 한 화면에서 비교할 수 있습니다.</sub></p>

GIF가 단일 run을 보여준다면, gallery는 seed에 따른 편차를 보여줍니다. 생성되는 Markdown report에는 best/worst seed의 storyboard와 metrics 미리보기가 본문에 포함됩니다. 전체 예시: [docs/media/make_moons_seed_report.md](./docs/media/make_moons_seed_report.md).

## T-WFC vs SGD+Momentum 비교

T-WFC는 각 weight를 5개 이산값(`{-1, -0.5, 0, 0.5, 1}`) 중 하나로 붕괴시키고, SGD는 연속 실수 weight를 최적화합니다. Baseline은 momentum(0.9)과 learning-rate decay를 적용한 표준 SGD이며, 의도적으로 약화시킨 설정이 아닙니다.

<p align="center">
  <img src="./docs/media/make_moons_twfc_vs_sgd.gif" alt="T-WFC vs SGD on make_moons" width="950">
</p>
<p align="center"><sub><code>make_moons</code>에서 T-WFC 붕괴 진행과 SGD+momentum baseline을 나란히 비교합니다.</sub></p>

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
      <strong>가장 가까운 케이스: make_moons</strong><br>
      T-WFC hard weights(이산 5값)는 test accuracy <code>0.925</code>, SGD+momentum(연속 실수)은 <code>1.000</code>까지 갑니다.
    </td>
    <td valign="top">
      <strong>확장 한계를 보여주는 케이스: spiral</strong><br>
      gap이 큽니다. T-WFC hard test accuracy는 <code>0.367</code>, SGD+momentum은 <code>0.911</code>입니다. 이산 제약과 단일 weight 붕괴 전략은 아직 이 복잡도까지 확장되지 못합니다.
    </td>
  </tr>
</table>

<p align="center">
  <img src="./docs/media/iris_twfc_vs_sgd_metrics.png" alt="T-WFC vs SGD metrics on iris" width="920">
</p>
<p align="center"><sub><code>iris</code>: T-WFC shadow test accuracy <code>0.750</code>, hard test accuracy <code>0.556</code>, SGD+momentum <code>0.944</code>.</sub></p>

`make_moons`가 T-WFC가 가장 유망해 보이는 케이스이고, `spiral`과 `iris`는 제대로 튜닝된 연속값 optimizer와의 솔직한 차이를 보여줍니다.

## 현재 상태

- 데이터셋: `make_moons`, `spiral`, vendored `iris.csv`.
- 모델: 단일 hidden layer toy MLP 및 다층 구성 (예: `2-24-24-3`).
- 학습 루프: 관측, 단일 weight 붕괴, 전파, rollback-aware backtracking, forced-commit fallback.
- 시각화: storyboard, GIF, metrics timeline, multi-seed gallery/report, `T-WFC vs SGD` 비교 board.
- `pyproject.toml`을 통해 설치형 `t-wfc` CLI를 제공합니다.
- 전체 기능 이력은 [CHANGELOG.md](./CHANGELOG.md)를 참고하세요.

## 빠른 시작

```bash
python3 -m pip install -e .
t-wfc --dataset make_moons --max-steps 8 --show-steps 6
t-wfc --dataset iris --hidden-layers 16,16 --max-steps 18 --compare-sgd --show-steps 6
```

시각화, seed sweep, stress test 옵션은 `t-wfc --help`를 참고하세요. 전체 실행 레시피 예시는 [docs/VERIFICATION.md](./docs/VERIFICATION.md)에 있습니다.

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
- `src/t_wfc/reporting.py`: inline highlight preview와 drill-down 링크가 포함된 Markdown seed-sweep report 생성
- `src/t_wfc/visualization.py`: overview, progress, metrics, storyboard, GIF, seed gallery, `T-WFC vs SGD` 비교 plot 생성
- `src/t_wfc/cli.py`: 커맨드라인 진입점
- `docs/media/`: 이 README에서 직접 사용하는 공개용 시각화 샘플 모음
- `pyproject.toml`: 패키지 메타데이터, 의존성, `t-wfc` console script

## 참고

- 아직 완성된 프레임워크가 아니라 연구용 프로토타입입니다.
- 런타임 핵심 의존성은 `numpy`입니다.
- 시각화 출력에는 `matplotlib`를 사용합니다.
- GIF export에는 `Pillow`를 사용합니다.
