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
- massive Klein-Gordon retarded Green function 의 light-cone shell 과 timelike tail 구조를 반영하는 `kg_retarded` 전파 모드 추가
- 구간별 중점 근사 대신 `quadrature_order`로 조절 가능한 Gauss-Legendre quadrature 를 써서 piecewise linear worldline 위의 연속 시간 위상 적분 정밀도 개선
- worldline pair 적분만 하지 않고 spacetime sample point 에서 `phi_rs`를 직접 평가하는 `sample_branch_field` 및 `compute_sampled_spacetime_phase` 추가
- target worldline 에서 `phi_rs` 표본을 직접 뽑는 `compute_phi_rs_samples` 추가
- 비등방 width tensor 를 쓰는 `compute_anisotropic_sampled_spacetime_phase` 추가
- 지정한 시공간 격자 위에서 장 표본을 직접 푸는 `FieldLattice`, `solve_field_lattice` 추가
- 격자 time slice 별 장 진화를 요약하는 `FieldEvolutionResult`, `solve_field_lattice_dynamics` 추가
- multiscale refinement 로 더 큰 격자 계층을 누적하는 `MultiscaleFieldEvolutionResult`, `solve_multiscale_field_lattice` 추가
- 경계조건과 간단한 spectral coefficient 요약을 주는 `SpectralLatticeResult`, `solve_spectral_lattice` 추가
- 시간 slice 별 경계조건 스케줄을 처리하는 `DynamicBoundaryLatticeResult`, `solve_dynamic_boundary_lattice` 추가
- branch matrix 로부터 qudit 얽힘에 대응하는 상대 위상 `compute_entanglement_phase` 구현
- branch pair 별 self-energy, 방향성 있는 cross-term, 대칭 interaction, total phase 를 분리해 주는 `analyze_branch_pair_phase` 구현
- branch 집합 전체에 대해 self phase 벡터, directed cross matrix, symmetric interaction matrix, total matrix 를 한 번에 계산하는 `analyze_phase_decomposition` 구현
- mediator 종류를 `scalar`/`vector`/`gravity` 로 나눠 계산하는 `compute_mediated_phase_matrix` 구현
- 여러 입자 worldline 묶음을 하나의 분기처럼 다루는 `CompositeBranch` 와 `compute_composite_phase_matrix` 구현
- isotropic Gaussian width 를 가진 branch 별 finite-width wavepacket 모델에 대해 Yukawa kernel 을 상대 반경 분포 위에서 직접 적분하는 `compute_wavepacket_phase_matrix` 구현
- finite-width wavepacket 모델에서도 self/cross/interaction/total 분해를 계산하는 `analyze_wavepacket_phase_decomposition` 구현
- 결과를 한 번에 묶는 `SimulationResult` 및 `simulate` 구현
- branch worldline 로부터 momentum mode 별 Fourier-space displacement amplitude `compute_branch_displacement_amplitudes` 구현
- 각운동량 방향 평균과 radial quadrature 를 써서 연속 운동량 적분을 근사하는 `compute_continuum_displacement_amplitudes` 구현
- 적응형 radial order refinement 를 쓰는 `compute_adaptive_continuum_displacement_amplitudes` 구현
- momentum shell 분할 기반의 `compute_split_continuum_displacement_amplitudes` 구현
- coarse/refined angular 평균을 비교해 오차를 추정하는 `AngularQuadratureResult`, `estimate_continuum_displacement_amplitudes` 구현
- 더 큰 angular basis 와 radial order 를 단계적으로 키우는 `ContinuumExtrapolationResult`, `compute_extrapolated_continuum_displacement_amplitudes` 구현
- extrapolated continuum 결과에서 절대/상대 오차 bound 를 추정하는 `SpectralErrorBoundResult`, `estimate_spectral_continuum_error_bound` 구현
- angular basis 확장에 따른 수렴 여부를 추적하는 `SpectralConvergenceResult`, `estimate_spectral_convergence` 구현
- 두 계의 branch 조합 `(r, s)`마다 displacement profile 을 합성하는 `compute_branch_pair_displacements` 구현
- displacement operator 합성에서 생기는 BCH 위상을 mode profile 로 계산하는 `compute_displacement_operator_phase` 구현
- branch pair 가 만드는 field coherent state 의 자유 진화와 occupation number 를 추적하는 `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state` 구현
- coherent-state vacuum suppression, overlap, 상대 위상을 함께 계산하는 `CoherentStateComparison`, `compare_coherent_states`, `analyze_branch_pair_coherent_overlap` 구현
- diagonal covariance 를 가진 Gaussian mode state overlap 과 coherent-state superposition overlap 을 위한 `GaussianModeState`, `ModeSuperpositionState`, `compare_gaussian_mode_states`, `compare_superposition_states` 구현
- 비대각 공분산 Gaussian overlap 과 cat-state overlap 을 위한 `GeneralGaussianState`, `CatModeState`, `compare_general_gaussian_states`, `compare_cat_mode_states` 구현
- mode sample 에서 Gaussian/cat 상태를 재구성하는 `tomograph_general_gaussian_state`, `tomograph_cat_mode_state` 구현
- 여러 branch/mode sample 묶음의 aggregate state 와 상대 위상 행렬을 복원하는 `MultimodeTomographyResult`, `tomograph_multimode_family` 구현
- branch label 이 붙은 sample 묶음을 symbolic bookkeeping 형태로 요약하는 `SymbolicBookkeepingResult`, `summarize_symbolic_multimode_bookkeeping` 구현
- multimode bookkeeping 에 대한 antisymmetry/norm consistency 를 점검하는 `AnalyticIdentityResult`, `verify_multimode_analytic_identities` 구현
- mediator field sample 을 직접 뽑는 `sample_mediator_field` 추가
- mediator-specific effective force 로 target worldline 을 업데이트하는 `evolve_backreacted_branch` 추가
- effective backreaction 을 여러 차례 반복 적용하는 `iterate_backreaction` 추가
- source/target 을 함께 갱신하는 `CoupledBackreactionResult`, `solve_coupled_backreaction` 추가
- update 크기에 따른 damping 을 거는 `NonlinearBackreactionResult`, `solve_nonlinear_backreaction` 추가
- residual 기준으로 반복을 멈추는 `SelfConsistentBackreactionResult`, `solve_self_consistent_backreaction` 추가
- 예제 실행용 `python -m relativistic_circuit_locality.demo` 추가 및 `instantaneous`/`retarded`/`time_symmetric`/`causal_history`/`kg_retarded` 위상 비교, branch pair phase 분해, finite-width wavepacket 위상, displacement/coherent-state 출력 지원
- `unittest` 기반 회귀 테스트 추가
- 서로 다른 시간 샘플링을 가진 branch 사이에서도 선형 보간 기반으로 거리, mediation, 위상을 계산하도록 개선
- Fourier-space displacement/coherent-state 계산에 대한 회귀 테스트 추가
- `causal_history`가 과거 source support 에 민감하고 past light cone overlap 이 없으면 0 이 되는지에 대한 회귀 테스트 추가
- `kg_retarded`가 massless limit 에서 `retarded`와 일치하고, massive field 에서 tail history 에 민감한지에 대한 회귀 테스트 추가
- 4차원 field-sampling 적분이 pointlike limit 과 finite-width 보정에 반응하는지에 대한 회귀 테스트 추가
- continuum displacement, coherent overlap, mediator variant, composite branch 합성에 대한 회귀 테스트 추가
- 비등방 field sampling, adaptive continuum quadrature, Gaussian/non-Gaussian overlap, mediator field sampling 에 대한 회귀 테스트 추가
- lattice field sampling, split continuum quadrature, general Gaussian/cat overlap, backreaction trajectory 에 대한 회귀 테스트 추가
- angular error estimate, mode-state tomography, lattice dynamics, iterative backreaction 에 대한 회귀 테스트 추가
- multiscale lattice, extrapolated continuum controller, multimode tomography, coupled backreaction 에 대한 회귀 테스트 추가
- spectral lattice boundary, spectral error bound, symbolic bookkeeping, nonlinear backreaction 에 대한 회귀 테스트 추가
- dynamic boundary schedule, spectral convergence, analytic identity check, self-consistent backreaction 에 대한 회귀 테스트 추가

## 현재 공개 API 요약

- 기본 기하/인과성: `BranchPath`, `TrajectoryPoint`, `compute_closest_approach`, `field_mediation_intervals`, `is_field_mediated`
- 위상 행렬: `compute_branch_phase_matrix`, `compute_entanglement_phase`, `simulate`
- 위상 분해: `analyze_branch_pair_phase`, `analyze_phase_decomposition`
- finite-width wavepacket: `compute_wavepacket_phase_matrix`, `analyze_wavepacket_phase_decomposition`
- Fourier/coherent-state: `compute_branch_displacement_amplitudes`, `compute_branch_pair_displacements`, `compute_displacement_operator_phase`, `CoherentStateEvolution`, `evolve_coherent_state`, `analyze_branch_pair_coherent_state`
- 직접 field sampling: `FieldSample`, `sample_branch_field`, `compute_sampled_spacetime_phase`, `compute_phi_rs_samples`
- 비등방 field sampling: `compute_anisotropic_sampled_spacetime_phase`
- lattice/backreaction: `FieldLattice`, `FieldEvolutionResult`, `MultiscaleFieldEvolutionResult`, `SpectralLatticeResult`, `DynamicBoundaryLatticeResult`, `solve_field_lattice`, `solve_field_lattice_dynamics`, `solve_multiscale_field_lattice`, `solve_spectral_lattice`, `solve_dynamic_boundary_lattice`, `evolve_backreacted_branch`, `iterate_backreaction`, `CoupledBackreactionResult`, `solve_coupled_backreaction`, `NonlinearBackreactionResult`, `solve_nonlinear_backreaction`, `SelfConsistentBackreactionResult`, `solve_self_consistent_backreaction`
- 연속 Fourier/coherent-state: `compute_continuum_displacement_amplitudes`, `compute_adaptive_continuum_displacement_amplitudes`, `compute_split_continuum_displacement_amplitudes`, `estimate_continuum_displacement_amplitudes`, `compute_extrapolated_continuum_displacement_amplitudes`, `estimate_spectral_continuum_error_bound`, `estimate_spectral_convergence`, `AngularQuadratureResult`, `ContinuumExtrapolationResult`, `SpectralErrorBoundResult`, `SpectralConvergenceResult`, `CoherentStateComparison`, `compare_coherent_states`, `GaussianModeState`, `GeneralGaussianState`, `ModeSuperpositionState`, `CatModeState`, `compare_gaussian_mode_states`, `compare_general_gaussian_states`, `compare_superposition_states`, `compare_cat_mode_states`, `tomograph_general_gaussian_state`, `tomograph_cat_mode_state`, `MultimodeTomographyResult`, `tomograph_multimode_family`, `SymbolicBookkeepingResult`, `summarize_symbolic_multimode_bookkeeping`, `AnalyticIdentityResult`, `verify_multimode_analytic_identities`, `analyze_branch_pair_coherent_overlap`
- 다입자/매개자 확장: `CompositeBranch`, `compute_mediated_phase_matrix`, `compute_composite_phase_matrix`, `sample_mediator_field`

## 전파 모드 요약

- `instantaneous`: 같은 시각의 branch 위치만 써서 Yukawa kernel 을 적분한다.
- `retarded`: target 시각마다 source 의 단일 retarded point 를 찾아 적분한다.
- `time_symmetric`: `A -> B`, `B -> A` retarded 기여를 평균해 방향성 비대칭을 줄인다.
- `causal_history`: past light cone 내부의 source history 전체를 proper-time Yukawa kernel 로 적분한다.
- `kg_retarded`: retarded shell `1/(4πr)`와 timelike tail `-m J1(mτ)/(4πτ)`를 함께 써서 massive Klein-Gordon retarded Green function 구조를 근사한다.

## 추가해야 할 기능

- sparse/FFT 기반의 실제 대규모 4차원 PDE solver 와 장 전체의 동적 경계조건 처리
- continuum momentum 적분의 provable error control 과 더 고차의 spectral basis
- Appendix D 전체 bookkeeping 을 analytic identity 와 상태 변환 수준까지 재현하는 완전한 multimode 상태 해석기
- gauge field/general relativity 의 실제 field equation solver 와 self-consistent nonlinear backreaction

## 현재 구현의 한계

- 기존의 "동시각 정적 상호작용만 본다"는 한계는 `retarded`, `time_symmetric`, `causal_history`, `kg_retarded` 전파 모드로 개선했다. 특히 `kg_retarded`는 massive Klein-Gordon retarded Green function 의 shell/tail 구조를 반영해 `phi_rs`를 더 직접적으로 근사한다. 다만 아직도 연속 4차원 source density 에 대한 완전한 retarded Green function 적분기라기보다, piecewise linear worldline 위에서 수치 적분하는 축약 모델이다.
- 기존의 "구간별 중점값 하나만 적분한다"는 한계는 Gauss-Legendre quadrature, 4차원 field-sampling 적분기, 비등방 width tensor sampling, 명시적 lattice sampling, time-slice lattice dynamics, multiscale lattice refinement, boundary-conditioned spectral lattice 요약, dynamic boundary schedule 로 개선했다. `compute_sampled_spacetime_phase`, `compute_anisotropic_sampled_spacetime_phase`, `solve_field_lattice`, `solve_field_lattice_dynamics`, `solve_multiscale_field_lattice`, `solve_spectral_lattice`, `solve_dynamic_boundary_lattice`는 target worldline 또는 지정한 시공간 격자 위에서 `phi_rs`를 직접 샘플링한다. 다만 아직도 full PDE solver 나 대규모 격자 진화 엔진은 아니다.
- self-energy 와 cross-term 분해는 추가했지만, 논문 식 (36)의 연속장 분해를 직접 푼 것이 아니라 현재 Yukawa 기반 수치 모델 위에서 해석한 것이다.
- finite-width wavepacket 은 isotropic Gaussian profile 로 모델링했고, 3차원 상대 반경 분포에 대한 수치 적분으로 처리한다. 아직 일반적인 비등방/비가우시안 packet 은 지원하지 않는다.
- continuum displacement 는 radial quadrature, 유한 개 각도 방향 평균, 적응형 refinement, momentum shell splitting, coarse/refined angular 오차 추정, extrapolated angular basis 확장, spectral-style error bound 추정, convergence flag 추적으로 연속 운동량 적분을 근사한다. 따라서 엄밀한 연속 Fourier 적분 전체를 해석적으로 푼 것은 아니다.
- coherent-state 비교는 vacuum suppression, diagonal/비대각 Gaussian covariance, finite coherent superposition, cat-state overlap, sample 기반 tomography, multimode aggregate phase matrix, symbolic bookkeeping 요약, analytic identity check 까지 포함하지만, Appendix D 의 모든 위상 bookkeeping 과 일반 상태 family 를 완전히 재현한 것은 아니다.
- 다입자와 gravity/vector 버전은 superposition, mediator-specific field sampling, effective backreaction, iterative backreaction, coupled backreaction, damped nonlinear backreaction, self-consistent residual stopping 수준의 일반화다. 실제 gauge/gravitational field equation solver 는 아니다.
- 구현 전체는 여전히 QFT full evolution 이 아니라 논문의 parametric approximation 안에서 움직이는 축약 모델이다.
- microcausality 자체를 commutator 적분으로 평가하지 않고, spacelike separation criterion 으로 판정한다.

## 사용 방법

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
PYTHONPATH=src python -m unittest discover -s tests -v
```
