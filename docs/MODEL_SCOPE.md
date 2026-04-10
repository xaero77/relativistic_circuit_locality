# Model Scope

## 구현 목적

이 저장소는 논문 `"Circuit locality from relativistic locality in scalar field mediated entanglement"`의 핵심 회로-국소성 아이디어를 계산 가능한 Python 참조 모델로 정리한 것이다. 목표는 개념 검증과 재현 가능한 수치 실험이지, 완전한 연속 QFT 시뮬레이터를 제공하는 것이 아니다.

## 구현 가정

- branch worldline 은 시간 이산화된 trajectory sample 또는 spline 보간으로 표현한다.
- 핵심 phase 계산은 Yukawa/retarded/causal-history/Klein-Gordon surrogate 커널을 사용한다.
- 다수의 확장 기능은 exact dynamics 가 아니라 연구용 surrogate bundle 형태로 제공된다.
- 설치 기반 실행을 기본으로 하며, 예제와 테스트는 editable install 상태를 기준으로 검증한다.

## 안정 범위

다음 기능은 패키지 루트에서 직접 접근하는 안정 surface 로 간주한다.

- branch geometry 와 spline interpolation
- spacelike mediation interval 판정
- core branch phase matrix
- finite-width wavepacket phase matrix
- entangling phase 요약
- 기본 `simulate` wrapper

## 실험 범위

다음 영역은 `experimental` namespace 아래의 확장 기능으로 본다.

- field sampling 과 sampled decomposition
- continuum/spectral displacement 및 Lebedev quadrature
- tomography, multimode bookkeeping, universal state bundles
- lattice/PDE/backreaction surrogate
- tensor mediator, renormalization, decoherence, multi-body correlation
- finite-difference KG, physical lattice dynamics, Fock-space/Magnus, running coupling

## 현재 한계

- 많은 고급 기능이 surrogate 모델이므로 exact QFT 해석과 동일하지 않다.
- `scalar_field.py`는 여전히 저장소에서 가장 큰 구현 모듈이며, 추가 책임 분리는 계속 가능한 상태다.
- Lebedev table 은 lazy-loading 으로 옮겼지만 데이터 파일 자체는 여전히 저장소에 vendor 된다.
- `ruff`/`mypy` 설정과 benchmark 진입점은 추가했지만, 자동 CI 파이프라인은 아직 없다.

## 권장 사용 방식

1. `relativistic_circuit_locality` 또는 `relativistic_circuit_locality.core`에서 안정 API 를 사용한다.
2. 추가 물리 모델이나 surrogate bundle 이 필요할 때만 `relativistic_circuit_locality.experimental`로 내려간다.
3. 재현 가능한 실행이 필요하면 `python -m relativistic_circuit_locality.experiments ...`로 spec 기반 batch 실행을 사용한다.
4. 회귀 검증은 `python -m unittest discover -s tests -v`를 기준으로 수행한다.
