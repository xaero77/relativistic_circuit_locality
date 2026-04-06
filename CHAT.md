# CHAT

## 논문 해석 기준

읽은 논문은 `q-2026-03-24-2046.pdf`이며, 구현은 다음 핵심 식과 주장에 맞춰 범위를 잡았다.

- 식 (5), (7), (72), (73): branch 별 source support 가 spacelike separated 이면 evolution 이 field mediation circuit 형태를 가진다.
- 식 (36): branch 위상 `theta_rs = -1/2 ∫ d^4x rho_rs(x) phi_rs(x)`.
- 식 (64): 각 branch 에서 양자 입자는 고전 source density `rho_A^r + rho_B^s`로 근사된다.
- 본문 및 Appendix A/B/C: 전체 논리는 parametric approximation 위에서 성립한다.

## 필요한 기능

- branch 별 입자 경로와 전하를 기술할 수 있는 데이터 구조
- branch 쌍의 최소 거리 `d_min` 계산
- 시간 구간별 spacelike separation 판정
- field mediation 이 성립하는 시간 구간 추출
- scalar-mediated phase `theta_rs` 계산
- 서로 다른 branch 조합에서 상대 얽힘 위상 계산
- 재현 가능한 예제와 테스트

## 구현된 기능

- `BranchPath`, `TrajectoryPoint` 데이터 구조 추가
- 논문 조건 `d_min > 0`에 대응하는 `compute_closest_approach` 구현
- 모든 branch 쌍이 spacelike 로 유지되는 시간 구간을 찾는 `field_mediation_intervals` 구현
- 전체 실험 구간이 mediation 조건을 만족하는지 판정하는 `is_field_mediated` 구현
- 논문 식 (36)을 직접 연속장으로 풀지 않고, 저에너지/준정적 수치 근사로 바꾼 Yukawa kernel 기반 `compute_branch_phase_matrix` 구현
- branch matrix 로부터 qudit 얽힘에 대응하는 상대 위상 `compute_entanglement_phase` 구현
- 결과를 한 번에 묶는 `SimulationResult` 및 `simulate` 구현
- 예제 실행용 `python -m relativistic_circuit_locality.demo` 추가
- `unittest` 기반 회귀 테스트 추가
- 서로 다른 시간 샘플링을 가진 branch 사이에서도 선형 보간 기반으로 거리, mediation, 위상을 계산하도록 개선

## 추가해야 할 기능

- 식 (84), (85), (90), (103)을 직접 따르는 Fourier-space displacement operator 수치 구현
- massive Klein-Gordon retarded solution 기반의 완전한 `phi_rs` 계산기
- source self-energy 와 cross-term 을 분리한 위상 분석 도구
- branch 별 compact support 대신 finite-width wavepacket 을 직접 적분하는 모델
- 논문 Appendix D 수준의 coherent-state 진화 추적
- 다입자 일반화와 gauge field/general relativity 버전

## 현재 구현의 한계

- QFT full evolution 이 아니라 논문의 parametric approximation 안에서 움직이는 축약 모델이다.
- 연속 시공간 적분을 직접 수행하지 않고, piecewise linear worldline 과 Yukawa 준정적 근사를 사용한다.
- microcausality 자체를 commutator 적분으로 평가하지 않고, spacelike separation criterion 으로 판정한다.

## 사용 방법

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
PYTHONPATH=src python -m unittest discover -s tests -v
```
