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
- 같은 시각 정적 상호작용만 보던 한계를 줄이기 위해 finite propagation speed 를 반영한 `retarded` 위상 적분 모드 추가
- 한 방향 `retarded` 근사의 비대칭을 줄이기 위해 `A -> B`, `B -> A` light-cone contribution 을 평균하는 `time_symmetric` 전파 모드 추가
- 단일 retarded point 대신 과거 light cone 전체를 적분하는 `causal_history` 전파 모드 추가
- 구간별 중점 근사 대신 `quadrature_order`로 조절 가능한 Gauss-Legendre quadrature 를 써서 piecewise linear worldline 위의 연속 시간 위상 적분 정밀도 개선
- branch matrix 로부터 qudit 얽힘에 대응하는 상대 위상 `compute_entanglement_phase` 구현
- branch pair 별 self-energy, 방향성 있는 cross-term, 대칭 interaction, total phase 를 분리해 주는 `analyze_branch_pair_phase` 구현
- branch 집합 전체에 대해 self phase 벡터, directed cross matrix, symmetric interaction matrix, total matrix 를 한 번에 계산하는 `analyze_phase_decomposition` 구현
- isotropic Gaussian width 를 가진 branch 별 finite-width wavepacket 모델에 대해 Yukawa kernel 을 상대 반경 분포 위에서 직접 적분하는 `compute_wavepacket_phase_matrix` 구현
- finite-width wavepacket 모델에서도 self/cross/interaction/total 분해를 계산하는 `analyze_wavepacket_phase_decomposition` 구현
- 결과를 한 번에 묶는 `SimulationResult` 및 `simulate` 구현
- branch worldline 로부터 momentum mode 별 Fourier-space displacement amplitude `compute_branch_displacement_amplitudes` 구현
- 두 계의 branch 조합 `(r, s)`마다 displacement profile 을 합성하는 `compute_branch_pair_displacements` 구현
- displacement operator 합성에서 생기는 BCH 위상을 mode profile 로 계산하는 `compute_displacement_operator_phase` 구현
- branch pair 가 만드는 field coherent state 의 자유 진화와 occupation number 를 추적하는 `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state` 구현
- 예제 실행용 `python -m relativistic_circuit_locality.demo` 추가 및 `instantaneous`/`retarded`/`time_symmetric`/`causal_history` 위상 비교, branch pair phase 분해, finite-width wavepacket 위상, displacement/coherent-state 출력 지원
- `unittest` 기반 회귀 테스트 추가
- 서로 다른 시간 샘플링을 가진 branch 사이에서도 선형 보간 기반으로 거리, mediation, 위상을 계산하도록 개선
- Fourier-space displacement/coherent-state 계산에 대한 회귀 테스트 추가
- `causal_history`가 과거 source support 에 민감하고 past light cone overlap 이 없으면 0 이 되는지에 대한 회귀 테스트 추가

## 현재 공개 API 요약

- 기본 기하/인과성: `BranchPath`, `TrajectoryPoint`, `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- 위상 행렬: `compute_branch_phase_matrix`, `compute_entanglement_phase`, `simulate`
- 위상 분해: `analyze_branch_pair_phase`, `analyze_phase_decomposition`
- finite-width wavepacket: `compute_wavepacket_phase_matrix`, `analyze_wavepacket_phase_decomposition`
- Fourier/coherent-state: `compute_branch_displacement_amplitudes`, `compute_branch_pair_displacements`, `compute_displacement_operator_phase`, `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state`

## 전파 모드 요약

- `instantaneous`: 같은 시각의 branch 위치만 써서 Yukawa kernel 을 적분한다.
- `retarded`: target 시각마다 source 의 단일 retarded point 를 찾아 적분한다.
- `time_symmetric`: `A -> B`, `B -> A` retarded 기여를 평균해 방향성 비대칭을 줄인다.
- `causal_history`: past light cone 내부의 source history 전체를 proper-time Yukawa kernel 로 적분한다.

## 추가해야 할 기능

- massive Klein-Gordon retarded solution 기반의 완전한 `phi_rs` 계산기
- 논문 식 (84), (85), (90), (103)의 연속 운동량 적분을 직접 재현하는 고정밀 Fourier-space displacement operator 적분기
- 논문 Appendix D 의 모든 위상 항과 vacuum overlap 까지 포함하는 완전한 coherent-state 진화 추적
- 다입자 일반화와 gauge field/general relativity 버전

## 현재 구현의 한계

- 기존의 "동시각 정적 상호작용만 본다"는 한계는 `retarded`, `time_symmetric`, `causal_history` 전파 모드로 개선했다. 특히 `causal_history`는 single-source retarded Yukawa 대신 과거 light cone 내부의 source history 전체를 proper-time Yukawa kernel 로 적분한다. 다만 여전히 massive Klein-Gordon retarded Green function 자체를 직접 적분하는 완전한 `phi_rs` 계산기는 아니다.
- 기존의 "구간별 중점값 하나만 적분한다"는 한계는 Gauss-Legendre quadrature 로 개선했다. 다만 전체 4차원 장 방정식을 직접 푸는 적분기는 아니다.
- self-energy 와 cross-term 분해는 추가했지만, 논문 식 (36)의 연속장 분해를 직접 푼 것이 아니라 현재 Yukawa 기반 수치 모델 위에서 해석한 것이다.
- finite-width wavepacket 은 isotropic Gaussian profile 로 모델링했고, 3차원 상대 반경 분포에 대한 수치 적분으로 처리한다. 아직 일반적인 비등방/비가우시안 packet 은 지원하지 않는다.
- 새 displacement/coherent-state 계산은 선택한 유한 개의 momentum mode 위에서 수치 적분한다. 따라서 논문 식의 연속 Fourier 적분 전체를 그대로 구현한 것은 아니다.
- coherent-state 진화는 자유장 위상 회전과 occupation number 추적까지는 포함하지만, Appendix D 의 모든 vacuum phase 및 overlap determinant 를 아직 포함하지 않는다.
- 구현 전체는 여전히 QFT full evolution 이 아니라 논문의 parametric approximation 안에서 움직이는 축약 모델이다.
- microcausality 자체를 commutator 적분으로 평가하지 않고, spacelike separation criterion 으로 판정한다.

## 사용 방법

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
PYTHONPATH=src python -m unittest discover -s tests -v
```
