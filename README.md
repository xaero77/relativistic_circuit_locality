# relativistic_circuit_locality

`q-2026-03-24-2046.pdf`
("Circuit locality from relativistic locality in scalar field mediated entanglement", arXiv:2305.05645v4)
논문의 핵심 아이디어를 Python으로 옮긴 참조 구현이다.

이 저장소는 완전한 QFT 시뮬레이터가 아니라, 논문에서 직접 계산 가능한 핵심 구성요소를 설치 가능한 `src/` 레이아웃 패키지로 정리한 참조 모델이다. 기본 진입점은 안정 API와 시나리오별 예제에 집중하고, 연구형 surrogate 및 상세 기능 목록은 별도 문서로 분리했다.

## 핵심 기능

- worldline branch 기하와 spacelike mediation 판정
- scalar-mediated phase matrix 와 branch entangling phase 계산
- finite-width wavepacket phase 와 sampled field decomposition
- lattice/backreaction surrogate 와 research-grade closure 묶음
- 설치 기반 실행과 주제별 예제, 분리된 회귀 테스트

## 빠른 시작

설치:

```bash
python -m pip install -e .
```

예제:

```bash
python -m relativistic_circuit_locality.examples core
python -m relativistic_circuit_locality.examples field
python -m relativistic_circuit_locality.examples research
```

테스트:

```bash
python -m unittest discover -s tests -v
```

## 안정 API

패키지 루트와 [`src/relativistic_circuit_locality/core.py`](src/relativistic_circuit_locality/core.py)는 다음 안정 진입점만 노출한다.

- `BranchPath`, `TrajectoryPoint`, `SplineBranchPath`
- `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- `compute_branch_phase_matrix`, `compute_wavepacket_phase_matrix`
- `compute_entanglement_phase`, `simulate`

연구형 또는 확장 기능은 `relativistic_circuit_locality.experimental`에서 접근한다.

## 문서 경로

- 상세 기능/계층 요약: [`docs/API_SURFACE.md`](docs/API_SURFACE.md)
- 구현 범위, surrogate 수준, 한계: [`docs/MODEL_SCOPE.md`](docs/MODEL_SCOPE.md)
- 구현 이력과 세부 메모: [`CHAT.md`](CHAT.md)
- 개선 과제 추적: [`IMPROVEMENTS.md`](IMPROVEMENTS.md)

## 저장소 구조

- `src/relativistic_circuit_locality/core.py`: 안정 API entrypoint
- `src/relativistic_circuit_locality/experimental.py`: 확장 및 연구형 API namespace
- `src/relativistic_circuit_locality/scalar_field.py`: 핵심 수치 모델 구현
- `src/relativistic_circuit_locality/geometry.py`: worldline/spline 기하 계층
- `src/relativistic_circuit_locality/examples/`: 시나리오별 실행 예제
- `tests/`: 주제별로 분리된 회귀 테스트
