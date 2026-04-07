# relativistic_circuit_locality

`q-2026-03-24-2046.pdf`
("Circuit locality from relativistic locality in scalar field mediated entanglement", arXiv:2305.05645v4)
논문의 핵심 아이디어를 Python으로 옮긴 참조 구현이다.

이 저장소는 논문 전체를 완전한 양자장론 시뮬레이터로 재현하려는 목적이 아니라, 논문에서 직접 계산 가능한 핵심 구성요소를 작고 의존성 없는 수치 모델로 구현하는 데 초점을 둔다.

구현 범위는 다음과 같다.

- 두 quantum-controlled source의 branch별 궤적 표현
- field mediation circuit 성립 여부를 위한 spacelike separation 판정
- 논문의 `d_min > 0` 조건에 대응하는 최소 접근 거리 계산
- on-shell scalar-field phase `theta_rs = -1/2 ∫ rho_rs phi_rs`의 준정적 Yukawa 근사
- 한 방향 retarded 근사와 이를 대칭화한 time-symmetric light-cone 평균 모드
- 과거 light cone 전체를 적분하는 causal-history 전파 모드
- massive Klein-Gordon retarded Green function 의 shell/tail 구조를 반영하는 kg_retarded 모드
- branch 조합 사이의 상대 얽힘 위상 계산
- self-energy, directed cross-term, symmetric interaction, total phase 분해
- isotropic Gaussian finite-width wavepacket 위상 계산
- 유한 개 momentum mode 에 대한 Fourier-space displacement amplitude 와 coherent-state 자유 진화 추적

현재 코드는 논문의 parametric approximation 안에서 동작한다. 즉, 완전한 QFT 동역학을 직접 적분하지 않고, 시간 이산화된 궤적과 준정적 상호작용 커널을 이용해 논문의 구조를 계산 가능한 형태로 단순화했다.

최근 개선으로 branch 들이 동일한 시간 샘플을 공유해야 한다는 제약을 없앴다. 서로 다른 시간 이산화로 주어진 궤적도 겹치는 시간 구간에서 선형 보간하여 최소 접근 거리, mediation interval, branch phase 를 계산한다.

## 공개 API

- 기하/인과성: `BranchPath`, `TrajectoryPoint`, `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- 위상 계산: `compute_branch_phase_matrix`, `compute_entanglement_phase`, `simulate`
- 위상 분해: `analyze_branch_pair_phase`, `analyze_phase_decomposition`
- wavepacket: `compute_wavepacket_phase_matrix`, `analyze_wavepacket_phase_decomposition`
- Fourier/coherent-state: `compute_branch_displacement_amplitudes`, `compute_branch_pair_displacements`, `compute_displacement_operator_phase`, `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state`

## 전파 모드

- `instantaneous`: 동시각 Yukawa 상호작용
- `retarded`: 단일 retarded source point 근사
- `time_symmetric`: 양방향 retarded 평균
- `causal_history`: past light cone 내부 source history 적분
- `kg_retarded`: retarded shell 과 timelike tail 을 함께 적분하는 massive Klein-Gordon 근사

## 파일 구성

- `src/relativistic_circuit_locality/scalar_field.py`: 핵심 모델과 수치 계산 함수
- `src/relativistic_circuit_locality/demo.py`: 최소 실행 예제
- `tests/test_scalar_field.py`: 단위 테스트
- `CHAT.md`: 논문 해석, 구현된 기능, 추가 필요 기능 정리

## 빠른 실행

예제 실행:

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
```

테스트 실행:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
