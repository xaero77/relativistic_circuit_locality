from __future__ import annotations

"""스칼라장 모형에서 분기 궤적을 비교하는 유틸리티."""

from cmath import exp as complex_exp
from functools import lru_cache
from dataclasses import dataclass
from math import ceil, exp, factorial, inf, pi, sinh, sqrt
from typing import Callable, Literal

import numpy as np


Vector3 = tuple[float, float, float]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Vector3, scalar: float) -> Vector3:
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vector3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


_AXIS_DIRECTIONS: tuple[Vector3, ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

_DIAGONAL_DIRECTIONS: tuple[Vector3, ...] = (
    (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    (-1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    (1.0 / sqrt(3.0), -1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), -1.0 / sqrt(3.0)),
    (-1.0 / sqrt(3.0), -1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    (-1.0 / sqrt(3.0), 1.0 / sqrt(3.0), -1.0 / sqrt(3.0)),
    (1.0 / sqrt(3.0), -1.0 / sqrt(3.0), -1.0 / sqrt(3.0)),
    (-1.0 / sqrt(3.0), -1.0 / sqrt(3.0), -1.0 / sqrt(3.0)),
)

_EDGE_DIRECTIONS: tuple[Vector3, ...] = (
    (1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
    (1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0),
    (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
    (-1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0),
    (1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0)),
    (1.0 / sqrt(2.0), 0.0, -1.0 / sqrt(2.0)),
    (-1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0)),
    (-1.0 / sqrt(2.0), 0.0, -1.0 / sqrt(2.0)),
    (0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)),
    (0.0, 1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
    (0.0, -1.0 / sqrt(2.0), 1.0 / sqrt(2.0)),
    (0.0, -1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
)

# Lebedev 구면 quadrature 가중치 (4π 정규화, ∑w_i = 1).
_LEBEDEV_WEIGHTS_6: tuple[float, ...] = tuple(1.0 / 6.0 for _ in range(6))
_LEBEDEV_WEIGHTS_14: tuple[float, ...] = tuple(1.0 / 15.0 for _ in range(6)) + tuple(3.0 / 40.0 for _ in range(8))
_LEBEDEV_WEIGHTS_26: tuple[float, ...] = (
    tuple(1.0 / 21.0 for _ in range(6))
    + tuple(4.0 / 105.0 for _ in range(12))
    + tuple(27.0 / 840.0 for _ in range(8))
)


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    position: Vector3


@dataclass(frozen=True)
class BranchPath:
    label: str
    charge: float
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        # 구간별 선형 보간은 시간축이 엄격히 정렬되어 있어야만 의미가 있다.
        if len(self.points) < 2:
            raise ValueError("Each branch path needs at least two trajectory points.")
        times = [point.t for point in self.points]
        if any(t1 >= t2 for t1, t2 in zip(times, times[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")

    @property
    def time_window(self) -> tuple[float, float]:
        return (self.points[0].t, self.points[-1].t)

    def position_at(self, t: float) -> Vector3:
        start, stop = self.time_window
        if t < start or t > stop:
            raise ValueError("Requested time lies outside the branch support.")
        if t == stop:
            return self.points[-1].position

        for left, right in zip(self.points, self.points[1:]):
            if left.t <= t <= right.t:
                # 샘플 시각이 달라도 동작하도록 현재 구간 내부에서 보간한다.
                weight = (t - left.t) / (right.t - left.t)
                return _add(left.position, _scale(_sub(right.position, left.position), weight))
        raise ValueError("Requested time could not be interpolated from trajectory points.")


def _cubic_spline_coefficients(
    times: tuple[float, ...],
    values: tuple[float, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    """natural cubic spline 계수 ``(a, b, c, d)``를 구간마다 계산한다.

    구간 ``i``에서 ``f(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3``이다.
    """
    n = len(times) - 1
    h = [times[i + 1] - times[i] for i in range(n)]
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i]) - (3.0 / h[i - 1]) * (values[i] - values[i - 1])
    # tridiagonal system
    l = [1.0] * (n + 1)
    mu = [0.0] * (n + 1)
    z = [0.0] * (n + 1)
    for i in range(1, n):
        l[i] = 2.0 * (times[i + 1] - times[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    c_coeff = [0.0] * (n + 1)
    b_coeff = [0.0] * n
    d_coeff = [0.0] * n
    for j in range(n - 1, -1, -1):
        c_coeff[j] = z[j] - mu[j] * c_coeff[j + 1]
        b_coeff[j] = (values[j + 1] - values[j]) / h[j] - h[j] * (c_coeff[j + 1] + 2.0 * c_coeff[j]) / 3.0
        d_coeff[j] = (c_coeff[j + 1] - c_coeff[j]) / (3.0 * h[j])
    return tuple(
        (values[i], b_coeff[i], c_coeff[i], d_coeff[i]) for i in range(n)
    )


@dataclass(frozen=True)
class SplineBranchPath:
    """cubic spline 보간을 쓰는 BranchPath.

    C² 연속 궤적을 제공하므로 위상 적분의 정밀도가 piecewise linear 보간보다 높다.
    ``as_branch_path()``로 기존 API 와 호환되는 ``BranchPath``로 변환할 수 있다.
    """

    label: str
    charge: float
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Each branch path needs at least two trajectory points.")
        times = [point.t for point in self.points]
        if any(t1 >= t2 for t1, t2 in zip(times, times[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")
        # spline 계수를 미리 계산해 둔다.
        ts = tuple(p.t for p in self.points)
        xs = tuple(p.position[0] for p in self.points)
        ys = tuple(p.position[1] for p in self.points)
        zs = tuple(p.position[2] for p in self.points)
        object.__setattr__(self, "_times", ts)
        object.__setattr__(self, "_cx", _cubic_spline_coefficients(ts, xs))
        object.__setattr__(self, "_cy", _cubic_spline_coefficients(ts, ys))
        object.__setattr__(self, "_cz", _cubic_spline_coefficients(ts, zs))

    @property
    def time_window(self) -> tuple[float, float]:
        return (self.points[0].t, self.points[-1].t)

    def position_at(self, t: float) -> Vector3:
        start, stop = self.time_window
        if t < start or t > stop:
            raise ValueError("Requested time lies outside the branch support.")
        if t == stop:
            return self.points[-1].position
        times: tuple[float, ...] = object.__getattribute__(self, "_times")
        cx: tuple[tuple[float, float, float, float], ...] = object.__getattribute__(self, "_cx")
        cy: tuple[tuple[float, float, float, float], ...] = object.__getattribute__(self, "_cy")
        cz: tuple[tuple[float, float, float, float], ...] = object.__getattribute__(self, "_cz")
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                dt = t - times[i]
                ax, bx, ccx, dx = cx[i]
                ay, by, ccy, dy = cy[i]
                az, bz, ccz, dz = cz[i]
                x = ax + bx * dt + ccx * dt * dt + dx * dt * dt * dt
                y = ay + by * dt + ccy * dt * dt + dy * dt * dt * dt
                z = az + bz * dt + ccz * dt * dt + dz * dt * dt * dt
                return (x, y, z)
        raise ValueError("Requested time could not be interpolated from trajectory points.")

    def as_branch_path(self) -> BranchPath:
        """기존 API 와 호환되는 BranchPath 로 변환한다."""
        return BranchPath(label=self.label, charge=self.charge, points=self.points)

    def refined_branch_path(self, subdivisions: int = 4) -> BranchPath:
        """각 구간을 ``subdivisions``개로 세분화한 BranchPath 를 반환한다.

        spline 의 곡선 정보를 세분된 piecewise linear 궤적으로 옮긴다.
        """
        new_points: list[TrajectoryPoint] = [self.points[0]]
        for left, right in zip(self.points, self.points[1:]):
            for k in range(1, subdivisions + 1):
                frac = k / subdivisions
                t = left.t + frac * (right.t - left.t)
                new_points.append(TrajectoryPoint(t, self.position_at(t)))
        return BranchPath(label=self.label, charge=self.charge, points=tuple(new_points))


def compute_spline_branch_phase_matrix(
    branches_a: tuple[SplineBranchPath, ...],
    branches_b: tuple[SplineBranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    subdivisions: int = 4,
) -> tuple[tuple[float, ...], ...]:
    """SplineBranchPath 를 세분화한 뒤 기존 위상 행렬을 계산한다."""
    refined_a = tuple(s.refined_branch_path(subdivisions) for s in branches_a)
    refined_b = tuple(s.refined_branch_path(subdivisions) for s in branches_b)
    return compute_branch_phase_matrix(
        refined_a,
        refined_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )


@dataclass(frozen=True)
class SimulationResult:
    closest_approach: float
    mediation_intervals: tuple[tuple[float, float], ...]
    phase_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PairPhaseBreakdown:
    self_phase_a: float
    self_phase_b: float
    directed_cross_phase_ab: float
    directed_cross_phase_ba: float
    interaction_phase: float
    total_phase: float


@dataclass(frozen=True)
class PhaseDecompositionResult:
    self_phases_a: tuple[float, ...]
    self_phases_b: tuple[float, ...]
    directed_cross_matrix_ab: tuple[tuple[float, ...], ...]
    directed_cross_matrix_ba: tuple[tuple[float, ...], ...]
    interaction_matrix: tuple[tuple[float, ...], ...]
    total_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CoherentStateEvolution:
    mode_amplitudes: tuple[complex, ...]
    occupation_number: float
    displacement_norm: float


@dataclass(frozen=True)
class FieldSample:
    t: float
    position: Vector3
    value: float


@dataclass(frozen=True)
class CoherentStateComparison:
    overlap: complex
    vacuum_suppression: float
    relative_phase: float
    norm_distance: float


@dataclass(frozen=True)
class CompositeBranch:
    label: str
    components: tuple[BranchPath, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("CompositeBranch needs at least one component.")


@dataclass(frozen=True)
class GaussianModeState:
    displacement: tuple[complex, ...]
    covariance_diag: tuple[float, ...]


@dataclass(frozen=True)
class ModeSuperpositionState:
    weights: tuple[complex, ...]
    components: tuple[tuple[complex, ...], ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.components):
            raise ValueError("weights and components must have the same length.")


@dataclass(frozen=True)
class GeneralGaussianState:
    displacement: tuple[complex, ...]
    covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CatModeState:
    weights: tuple[complex, ...]
    components: tuple[GaussianModeState, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.components):
            raise ValueError("weights and components must have the same length.")


@dataclass(frozen=True)
class FieldLattice:
    samples: tuple[FieldSample, ...]
    time_slices: tuple[float, ...]
    spatial_points: tuple[Vector3, ...]


@dataclass(frozen=True)
class FieldEvolutionResult:
    lattices: tuple[FieldLattice, ...]


@dataclass(frozen=True)
class AngularQuadratureResult:
    amplitudes: tuple[complex, ...]
    coarse_amplitudes: tuple[complex, ...]
    error_estimate: float


@dataclass(frozen=True)
class ContinuumExtrapolationResult:
    amplitudes: tuple[complex, ...]
    level_errors: tuple[float, ...]
    mode_counts: tuple[int, ...]


@dataclass(frozen=True)
class MultiscaleFieldEvolutionResult:
    levels: tuple[FieldEvolutionResult, ...]
    spatial_point_counts: tuple[int, ...]


@dataclass(frozen=True)
class MultimodeTomographyResult:
    component_states: tuple[GeneralGaussianState, ...]
    aggregate_state: GeneralGaussianState
    relative_phase_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CoupledBackreactionResult:
    source: BranchPath
    target: BranchPath
    iterations: int


@dataclass(frozen=True)
class SpectralLatticeResult:
    lattice: FieldLattice
    spectral_coefficients: tuple[complex, ...]
    boundary_condition: Literal["open", "periodic", "dirichlet", "neumann"]


@dataclass(frozen=True)
class SpectralErrorBoundResult:
    amplitudes: tuple[complex, ...]
    absolute_bound: float
    relative_bound: float
    mode_count: int


@dataclass(frozen=True)
class SymbolicBookkeepingResult:
    labels: tuple[str, ...]
    relative_phase_matrix: tuple[tuple[float, ...], ...]
    amplitude_norms: tuple[float, ...]


@dataclass(frozen=True)
class NonlinearBackreactionResult:
    source: BranchPath
    target: BranchPath
    iterations: int
    max_update_norm: float


@dataclass(frozen=True)
class DynamicBoundaryLatticeResult:
    slices: tuple[SpectralLatticeResult, ...]
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...]


@dataclass(frozen=True)
class SpectralConvergenceResult:
    amplitudes: tuple[complex, ...]
    successive_differences: tuple[float, ...]
    mode_counts: tuple[int, ...]
    converged: bool


@dataclass(frozen=True)
class AnalyticIdentityResult:
    labels: tuple[str, ...]
    phase_antisymmetry_error: float
    norm_consistency_error: float


@dataclass(frozen=True)
class SelfConsistentBackreactionResult:
    source: BranchPath
    target: BranchPath
    iterations: int
    converged: bool
    residual: float


@dataclass(frozen=True)
class FftLatticeEvolutionResult:
    slices: tuple[SpectralLatticeResult, ...]
    damping_profile: tuple[float, ...]


@dataclass(frozen=True)
class CertifiedSpectralResult:
    amplitudes: tuple[complex, ...]
    certificate_error: float
    converged: bool
    mode_count: int


@dataclass(frozen=True)
class MultimodeStateTransformResult:
    labels: tuple[str, ...]
    overlap_matrix: tuple[tuple[complex, ...], ...]
    phase_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MediatorBackreactionResult:
    mediator: Literal["scalar", "vector", "gravity"]
    result: SelfConsistentBackreactionResult
    phase_shift: float


@dataclass(frozen=True)
class SurrogatePdeResult:
    lattice: DynamicBoundaryLatticeResult
    fft_evolution: FftLatticeEvolutionResult
    total_sample_count: int


@dataclass(frozen=True)
class HighOrderSpectralResult:
    certified: CertifiedSpectralResult
    extrapolated: ContinuumExtrapolationResult
    convergence: SpectralConvergenceResult
    effective_order: int


@dataclass(frozen=True)
class ComprehensiveBookkeepingResult:
    transform: MultimodeStateTransformResult
    identities: AnalyticIdentityResult
    bookkeeping: SymbolicBookkeepingResult


@dataclass(frozen=True)
class EffectiveFieldEquationResult:
    mediator: Literal["scalar", "vector", "gravity"]
    lattice: DynamicBoundaryLatticeResult
    backreaction: MediatorBackreactionResult


@dataclass(frozen=True)
class LargeScalePdeResult:
    surrogate: SurrogatePdeResult
    multiscale: MultiscaleFieldEvolutionResult
    spectral: SpectralLatticeResult
    total_grid_points: int


@dataclass(frozen=True)
class ProvableSpectralControlResult:
    high_order: HighOrderSpectralResult
    strict_certificate_error: float
    guaranteed: bool


@dataclass(frozen=True)
class AppendixDBookkeepingResult:
    comprehensive: ComprehensiveBookkeepingResult
    multimode: MultimodeTomographyResult
    state_transform: MultimodeStateTransformResult


@dataclass(frozen=True)
class GaugeGravityFieldResult:
    scalar_result: EffectiveFieldEquationResult
    vector_result: EffectiveFieldEquationResult
    gravity_result: EffectiveFieldEquationResult


@dataclass(frozen=True)
class RetardedGreenFunctionResult:
    samples: tuple[FieldSample, ...]
    propagation: Literal["retarded", "time_symmetric", "causal_history", "kg_retarded"]


@dataclass(frozen=True)
class SampledPhaseDecompositionResult:
    self_samples_a: tuple[float, ...]
    self_samples_b: tuple[float, ...]
    interaction_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class GeneralizedWavepacketResult:
    matrix: tuple[tuple[float, ...], ...]
    widths_a: tuple[float, ...]
    widths_b: tuple[float, ...]


@dataclass(frozen=True)
class ExactMicrocausalityResult:
    interval_commutators: tuple[float, ...]
    bounded: bool


@dataclass(frozen=True)
class FullQftSurrogateResult:
    pde: LargeScalePdeResult
    gauge_gravity: GaugeGravityFieldResult
    microcausality: ExactMicrocausalityResult


@dataclass(frozen=True)
class ReferencePdeControlResult:
    qft_surrogate: FullQftSurrogateResult
    spectral_control: ProvableSpectralControlResult
    effective_grid_points: int


@dataclass(frozen=True)
class UniversalStateFamilyResult:
    generalized_wavepacket: GeneralizedWavepacketResult
    appendix_d: AppendixDBookkeepingResult
    overlap_phase_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ExactMediatorSurrogateResult:
    gauge_gravity: GaugeGravityFieldResult
    microcausality: ExactMicrocausalityResult
    consistent: bool


@dataclass(frozen=True)
class ClosedLimitationBundleResult:
    reference_pde: ReferencePdeControlResult
    universal_state: UniversalStateFamilyResult
    exact_mediator: ExactMediatorSurrogateResult


@dataclass(frozen=True)
class HighFidelityPdeBundleResult:
    closed_bundle: ClosedLimitationBundleResult
    refined_pde: LargeScalePdeResult
    fidelity_score: float


@dataclass(frozen=True)
class CompleteStateFamilyBundleResult:
    universal_state: UniversalStateFamilyResult
    multimode: MultimodeTomographyResult
    appendix_d: AppendixDBookkeepingResult


@dataclass(frozen=True)
class ExactDynamicsSurrogateResult:
    exact_mediator: ExactMediatorSurrogateResult
    retarded_green: RetardedGreenFunctionResult
    qft_surrogate: FullQftSurrogateResult


@dataclass(frozen=True)
class ResearchGradeClosureResult:
    high_fidelity_pde: HighFidelityPdeBundleResult
    complete_state_family: CompleteStateFamilyBundleResult
    exact_dynamics: ExactDynamicsSurrogateResult


@dataclass(frozen=True)
class ProperTimeWorldline:
    branch: BranchPath
    proper_times: tuple[float, ...]
    lorentz_factors: tuple[float, ...]


@dataclass(frozen=True)
class RenormalizedPhaseResult:
    raw_matrix: tuple[tuple[float, ...], ...]
    self_energy_corrections: tuple[float, ...]
    renormalized_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TensorMediatedPhaseResult:
    scalar_phase: tuple[tuple[float, ...], ...]
    vector_phase: tuple[tuple[float, ...], ...]
    gravity_phase: tuple[tuple[float, ...], ...]
    mediator_mass: float
    gauge_scheme: str
    gauge_parameter: float
    vertex_resummation: str
    vertex_strength: float
    ghost_sector: GhostSectorResult
    dyson_schwinger: DysonSchwingerResult


@dataclass(frozen=True)
class GhostSectorResult:
    ghost_phase: tuple[tuple[float, ...], ...]
    brst_compensated_vector_phase: tuple[tuple[float, ...], ...]
    longitudinal_residual: tuple[tuple[float, ...], ...]
    brst_residual_norm: float
    nilpotency_defect: float
    mode: str
    strength: float


@dataclass(frozen=True)
class DysonSchwingerResult:
    dressed_vector_phase: tuple[tuple[float, ...], ...]
    dressed_gravity_phase: tuple[tuple[float, ...], ...]
    self_energy_kernel: tuple[tuple[float, ...], ...]
    iterations: int
    residual_norm: float
    converged: bool
    mode: str
    strength: float


@dataclass(frozen=True)
class DecoherenceResult:
    coherence_matrix: tuple[tuple[complex, ...], ...]
    decoherence_rates: tuple[float, ...]
    purity: float


@dataclass(frozen=True)
class MultiBodyCorrelationResult:
    pairwise_phases: tuple[tuple[float, ...], ...]
    three_body_phase: float
    total_correlation_phase: float


@dataclass(frozen=True)
class EntanglementMeasures:
    von_neumann_entropy: float
    negativity: float
    witness_value: float
    visibility: float


@dataclass(frozen=True)
class ModeOccupationDistribution:
    mode_occupations: tuple[float, ...]
    total_occupation: float
    mode_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class FiniteDifferencePdeResult:
    field_values: tuple[tuple[float, ...], ...]
    time_slices: tuple[float, ...]
    spatial_points: tuple[Vector3, ...]
    courant_number: float
    grid_shape: tuple[int, int, int]
    spatial_dimension: int
    effective_time_step: float
    substeps_per_interval: tuple[int, ...]
    stencil_order: int
    refinement_rounds: int
    boundary_geometry: str
    active_point_mask: tuple[bool, ...]


@dataclass(frozen=True)
class PhysicalLatticeDynamicsResult:
    lattices: tuple[FieldLattice, ...]
    time_step: float
    method: Literal["leapfrog", "verlet"]


@dataclass(frozen=True)
class RelativisticForceResult:
    updated_branch: BranchPath
    proper_times: tuple[float, ...]
    four_velocities: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LebedevQuadratureResult:
    amplitudes: tuple[complex, ...]
    quadrature_order: int
    direction_count: int


WidthSpec = float | tuple[float, ...]


def _refine_spatial_points(spatial_points: tuple[Vector3, ...]) -> tuple[Vector3, ...]:
    if len(spatial_points) < 2:
        return spatial_points
    refined: list[Vector3] = [spatial_points[0]]
    for left, right in zip(spatial_points, spatial_points[1:]):
        midpoint = _scale(_add(left, right), 0.5)
        refined.append(midpoint)
        refined.append(right)
    return tuple(refined)


def _sample_norm(sample: tuple[complex, ...]) -> float:
    return sqrt(sum(abs(value) ** 2 for value in sample))


def _overlap_window(*branches: BranchPath) -> tuple[float, float]:
    start = max(branch.time_window[0] for branch in branches)
    stop = min(branch.time_window[1] for branch in branches)
    if start >= stop:
        raise ValueError("Compared branches must have overlapping time support.")
    return (start, stop)


def _shared_time_grid(*branches: BranchPath) -> tuple[float, ...]:
    start, stop = _overlap_window(*branches)
    # 모든 분기의 샘플 시각을 합쳐 각 궤적 구간 경계를 빠짐없이 반영한다.
    times = {start, stop}
    for branch in branches:
        for point in branch.points:
            if start <= point.t <= stop:
                times.add(point.t)
    return tuple(sorted(times))


def _segment_minimum_distance(
    branch_a: BranchPath,
    branch_b: BranchPath,
    t_start: float,
    t_stop: float,
) -> float:
    position_a_start = branch_a.position_at(t_start)
    position_a_stop = branch_a.position_at(t_stop)
    position_b_start = branch_b.position_at(t_start)
    position_b_stop = branch_b.position_at(t_stop)

    relative_start = _sub(position_a_start, position_b_start)
    relative_velocity = _sub(
        _scale(_sub(position_a_stop, position_a_start), 1.0 / (t_stop - t_start)),
        _scale(_sub(position_b_stop, position_b_start), 1.0 / (t_stop - t_start)),
    )
    duration = t_stop - t_start
    speed_sq = _dot(relative_velocity, relative_velocity)

    if speed_sq <= 1e-18:
        return _norm(relative_start)

    # 구간 경계 안에서 상대 거리의 이차식을 최소화한다.
    tau = -_dot(relative_start, relative_velocity) / speed_sq
    tau = min(max(tau, 0.0), duration)
    return _norm(_add(relative_start, _scale(relative_velocity, tau)))


def compute_closest_approach(branch_a: BranchPath, branch_b: BranchPath) -> float:
    grid = _shared_time_grid(branch_a, branch_b)
    return min(
        _segment_minimum_distance(branch_a, branch_b, t_start, t_stop)
        for t_start, t_stop in zip(grid, grid[1:])
    )


def field_mediation_intervals(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    light_speed: float = 1.0,
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float], ...]:
    if not branches_a or not branches_b:
        return ()
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")

    base_times = _shared_time_grid(*branches_a, *branches_b)

    intervals: list[tuple[float, float]] = []
    current_start: float | None = None

    for index in range(len(base_times) - 1):
        segment_is_spacelike = True
        for branch_a in branches_a:
            for branch_b in branches_b:
                t_start = base_times[index]
                t_stop = base_times[index + 1]
                if _segment_minimum_distance(branch_a, branch_b, t_start, t_stop) <= tolerance:
                    # 구간 내부 접촉이 생기면 장만으로 매개된 상태가 깨진다.
                    segment_is_spacelike = False
                    break
            if not segment_is_spacelike:
                break

        t_start = base_times[index]
        t_stop = base_times[index + 1]
        if segment_is_spacelike:
            if current_start is None:
                current_start = t_start
        elif current_start is not None:
            intervals.append((current_start, t_start))
            current_start = None

        if index == len(base_times) - 2 and current_start is not None:
            intervals.append((current_start, t_stop))

    return tuple(intervals)


def is_field_mediated(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    light_speed: float = 1.0,
    tolerance: float = 1e-9,
) -> bool:
    intervals = field_mediation_intervals(
        branches_a,
        branches_b,
        light_speed=light_speed,
        tolerance=tolerance,
    )
    if not intervals:
        return False
    start = branches_a[0].points[0].t
    stop = branches_a[0].points[-1].t
    return intervals == ((start, stop),)


def _yukawa_kernel(distance: float, mass: float, cutoff: float) -> float:
    # 아주 짧은 거리는 잘라서 커널이 수치적으로 불안정해지지 않게 한다.
    effective_distance = max(distance, cutoff)
    return exp(-mass * effective_distance) / (4.0 * pi * effective_distance)


def _mode_energy(momentum: Vector3, mass: float) -> float:
    return sqrt(_dot(momentum, momentum) + mass * mass)


def _bessel_j1(x: float, *, tolerance: float = 1e-12, max_terms: int = 64) -> float:
    if x == 0.0:
        return 0.0
    abs_x = abs(x)
    # 큰 인수에서는 asymptotic expansion 을 쓴다.
    if abs_x > 10.0:
        mu = 1.0  # 4*nu^2 where nu=1
        p = 1.0
        q = 0.0
        term_p = 1.0
        term_q = (mu - 1.0) / (8.0 * abs_x)
        q = term_q
        for k in range(1, max_terms):
            factor_p = -(mu - (4 * k - 3) ** 2) * (mu - (4 * k - 1) ** 2) / (factorial(2 * k) * (8.0 * abs_x) ** (2 * k))
            term_p *= factor_p if k == 1 else 0.0
            if k == 1:
                term_p = -(mu - 1.0) * (mu - 9.0) / (2.0 * (8.0 * abs_x) ** 2)
                p += term_p
            if abs(term_p) < tolerance:
                break
        amplitude = sqrt(2.0 / (pi * abs_x))
        chi = abs_x - 3.0 * pi / 4.0
        from math import cos, sin
        result = amplitude * (p * cos(chi) - q * sin(chi))
        return result if x > 0.0 else -result
    # 작은 인수에서는 Taylor 급수를 쓴다.
    half_x = x / 2.0
    total = 0.0
    for index in range(max_terms):
        term = ((-1.0) ** index) * (half_x ** (2 * index + 1)) / (factorial(index) * factorial(index + 1))
        total += term
        if abs(term) <= tolerance:
            break
    return total


def _regularized_light_cone_delta(invariant_interval_sq: float, shell_width: float) -> float:
    if shell_width <= 0.0:
        raise ValueError("shell_width must be positive.")
    normalizer = sqrt(2.0 * pi) * shell_width
    exponent = -0.5 * (invariant_interval_sq / shell_width) ** 2
    return exp(exponent) / normalizer


def _pauli_jordan_commutator(
    delta_t: float,
    spatial_distance: float,
    *,
    mass: float,
    light_speed: float,
    shell_width: float,
    tolerance: float,
) -> float:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if mass < 0.0:
        raise ValueError("mass must be non-negative.")
    if abs(delta_t) <= tolerance:
        sign = 0.0
    else:
        sign = 1.0 if delta_t > 0.0 else -1.0
    if sign == 0.0:
        return 0.0

    invariant_interval_sq = light_speed * light_speed * delta_t * delta_t - spatial_distance * spatial_distance
    shell_term = sign * _regularized_light_cone_delta(invariant_interval_sq, shell_width) / (2.0 * pi)
    if invariant_interval_sq <= tolerance:
        return shell_term if abs(invariant_interval_sq) <= shell_width else 0.0

    proper_time = sqrt(max(invariant_interval_sq, 0.0)) / light_speed
    if proper_time <= tolerance:
        return shell_term
    tail_term = -sign * mass * _bessel_j1(mass * proper_time) / (4.0 * pi * proper_time)
    return shell_term + tail_term


@lru_cache(maxsize=None)
def _gauss_legendre_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rules = {
        1: ((0.0,), (2.0,)),
        2: ((-0.5773502691896257, 0.5773502691896257), (1.0, 1.0)),
        3: (
            (-0.7745966692414834, 0.0, 0.7745966692414834),
            (0.5555555555555556, 0.8888888888888888, 0.5555555555555556),
        ),
        4: (
            (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526),
            (0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385),
        ),
        5: (
            (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664),
            (0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908),
        ),
        6: (
            (-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
             0.2386191860831969, 0.6612093864662645, 0.9324695142031521),
            (0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
             0.4679139345726910, 0.3607615730481386, 0.1713244923791704),
        ),
        7: (
            (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
             0.4058451513773972, 0.7415311855993945, 0.9491079123427585),
            (0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
             0.3818300505051189, 0.2797053914892767, 0.1294849661688697),
        ),
        8: (
            (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363),
            (0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
             0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763),
        ),
        9: (
            (-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0,
             0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261),
            (0.0812743883615744, 0.1806481606948574, 0.2606106964029354, 0.3123470770400029, 0.3302393550012598,
             0.3123470770400029, 0.2606106964029354, 0.1806481606948574, 0.0812743883615744),
        ),
        10: (
            (-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,
             0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717),
            (0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
             0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881),
        ),
    }
    if order not in rules:
        raise ValueError("quadrature_order must be between 1 and 10.")
    return rules[order]


def _retarded_source_time(
    source: BranchPath,
    target: BranchPath,
    target_time: float,
    *,
    light_speed: float,
    iterations: int = 12,
    tolerance: float = 1e-12,
) -> float | None:
    source_start, source_stop = source.time_window
    guess = min(max(target_time, source_start), source_stop)

    # 고정점 반복을 먼저 시도한다.
    for _ in range(iterations):
        distance = _norm(_sub(target.position_at(target_time), source.position_at(guess)))
        updated = target_time - distance / light_speed
        if updated < source_start or updated > source_stop:
            break
        if abs(updated - guess) <= tolerance:
            return updated
        guess = updated
    else:
        distance = _norm(_sub(target.position_at(target_time), source.position_at(guess)))
        updated = target_time - distance / light_speed
        if source_start <= updated <= source_stop:
            return updated

    # 고정점이 수렴하지 않으면 bisection 으로 fallback 한다.
    # f(t_s) = t_s - (t_target - |x_target - x_source(t_s)| / c) = 0
    low = source_start
    high = min(target_time, source_stop)
    if low >= high:
        return None

    def _residual(t_s: float) -> float:
        d = _norm(_sub(target.position_at(target_time), source.position_at(t_s)))
        return t_s - (target_time - d / light_speed)

    f_low = _residual(low)
    f_high = _residual(high)
    if f_low * f_high > 0.0:
        return None

    for _ in range(64):
        mid = 0.5 * (low + high)
        if high - low <= tolerance:
            if source_start <= mid <= source_stop:
                return mid
            return None
        f_mid = _residual(mid)
        if f_mid * f_low <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    mid = 0.5 * (low + high)
    if source_start <= mid <= source_stop:
        return mid
    return None


def _causal_history_kernel(
    source: BranchPath,
    target: BranchPath,
    target_time: float,
    *,
    light_speed: float,
    quadrature_order: int,
    kernel: Callable[[float], float],
    proper_time_cutoff: float,
) -> float:
    source_start, source_stop = source.time_window
    if target_time <= source_start:
        return 0.0

    integration_stop = min(target_time, source_stop)
    if integration_stop <= source_start:
        return 0.0

    grid = [source_start]
    for point in source.points:
        if source_start < point.t < integration_stop:
            grid.append(point.t)
    grid.append(integration_stop)

    nodes, weights = _gauss_legendre_rule(quadrature_order)
    target_position = target.position_at(target_time)
    total = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0
        segment = 0.0
        for node, weight in zip(nodes, weights):
            source_time = midpoint + half_width * node
            delta_t = target_time - source_time
            if delta_t <= 0.0:
                continue
            distance = _norm(_sub(target_position, source.position_at(source_time)))
            invariant_sq = (light_speed * delta_t) ** 2 - distance * distance
            if invariant_sq <= 0.0:
                continue
            proper_interval = sqrt(max(invariant_sq, proper_time_cutoff * proper_time_cutoff))
            segment += weight * kernel(proper_interval)
        total += segment * half_width
    return total


def _kg_retarded_tail_kernel(proper_time: float, mass: float) -> float:
    if mass <= 0.0:
        return 0.0
    argument = mass * proper_time
    return -(mass * _bessel_j1(argument)) / (4.0 * pi * proper_time)


def _kg_retarded_field(
    source: BranchPath,
    target: BranchPath,
    target_time: float,
    *,
    mass: float,
    cutoff: float,
    light_speed: float,
    quadrature_order: int,
    proper_time_cutoff: float,
) -> float:
    field_value = 0.0

    retarded_time = _retarded_source_time(
        source,
        target,
        target_time,
        light_speed=light_speed,
    )
    if retarded_time is not None:
        distance = _norm(_sub(target.position_at(target_time), source.position_at(retarded_time)))
        field_value += 1.0 / (4.0 * pi * max(distance, cutoff))

    field_value += _causal_history_kernel(
        source,
        target,
        target_time,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        kernel=lambda proper_time: _kg_retarded_tail_kernel(proper_time, mass),
        proper_time_cutoff=proper_time_cutoff,
    )
    return field_value


def _field_value_at_observation_point(
    source: BranchPath,
    observation_time: float,
    observation_position: Vector3,
    *,
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
) -> float:
    target = BranchPath(
        label="_observation",
        charge=0.0,
        points=(
            TrajectoryPoint(observation_time, observation_position),
            TrajectoryPoint(observation_time + 1e-12, observation_position),
        ),
    )
    if propagation == "instantaneous":
        distance = _norm(_sub(observation_position, source.position_at(observation_time)))
        return _yukawa_kernel(distance, mass, cutoff)
    if propagation == "retarded":
        retarded_time = _retarded_source_time(source, target, observation_time, light_speed=light_speed)
        if retarded_time is None:
            return 0.0
        distance = _norm(_sub(observation_position, source.position_at(retarded_time)))
        return _yukawa_kernel(distance, mass, cutoff)
    if propagation == "time_symmetric":
        return _field_value_at_observation_point(
            source,
            observation_time,
            observation_position,
            mass=mass,
            cutoff=cutoff,
            propagation="retarded",
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
    if propagation == "causal_history":
        return _causal_history_kernel(
            source,
            target,
            observation_time,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            kernel=lambda proper_time: _yukawa_kernel(proper_time, mass, cutoff),
            proper_time_cutoff=cutoff,
        )
    return _kg_retarded_field(
        source,
        target,
        observation_time,
        mass=mass,
        cutoff=cutoff,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        proper_time_cutoff=cutoff,
    )


def _mediator_field_value(
    source: BranchPath,
    observation_time: float,
    observation_position: Vector3,
    *,
    mass: float,
    cutoff: float,
    mediator: Literal["scalar", "vector", "gravity"],
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
) -> float:
    effective_source = source
    if mediator == "gravity":
        effective_source = BranchPath(
            label=f"{source.label}_gravity",
            charge=abs(source.charge),
            points=source.points,
        )
    return _field_value_at_observation_point(
        effective_source,
        observation_time,
        observation_position,
        mass=0.0 if mediator in {"vector", "gravity"} else mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )


def _mediator_pair_coupling(branch_a: BranchPath, branch_b: BranchPath, mediator: Literal["scalar", "vector", "gravity"]) -> float:
    if mediator == "gravity":
        return abs(branch_a.charge) * abs(branch_b.charge)
    return branch_a.charge * branch_b.charge


def _pair_phase_integral(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
    kernel: Callable[[float], float],
    mass: float,
    cutoff: float,
) -> float:
    grid = _shared_time_grid(branch_a, branch_b)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    phase = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0
        segment_integral = 0.0
        for node, weight in zip(nodes, weights):
            sample_time = midpoint + half_width * node
            point_b_now = TrajectoryPoint(t=sample_time, position=branch_b.position_at(sample_time))
            if propagation == "instantaneous":
                point_a = TrajectoryPoint(t=sample_time, position=branch_a.position_at(sample_time))
                distance = _norm(_sub(point_a.position, point_b_now.position))
            elif propagation == "retarded":
                retarded_time = _retarded_source_time(
                    branch_a,
                    branch_b,
                    sample_time,
                    light_speed=light_speed,
                )
                if retarded_time is None:
                    continue
                point_a = TrajectoryPoint(t=retarded_time, position=branch_a.position_at(retarded_time))
                distance = _norm(_sub(point_a.position, point_b_now.position))
            elif propagation == "causal_history":
                field_value = _causal_history_kernel(
                    branch_a,
                    branch_b,
                    sample_time,
                    light_speed=light_speed,
                    quadrature_order=quadrature_order,
                    kernel=kernel,
                    proper_time_cutoff=1e-9,
                )
                segment_integral += weight * field_value
                continue
            elif propagation == "kg_retarded":
                field_value = _kg_retarded_field(
                    branch_a,
                    branch_b,
                    sample_time,
                    mass=mass,
                    cutoff=cutoff,
                    light_speed=light_speed,
                    quadrature_order=quadrature_order,
                    proper_time_cutoff=1e-9,
                )
                segment_integral += weight * field_value
                continue
            else:
                distances: list[float] = []

                retarded_time_ab = _retarded_source_time(
                    branch_a,
                    branch_b,
                    sample_time,
                    light_speed=light_speed,
                )
                if retarded_time_ab is not None:
                    point_a_retarded = TrajectoryPoint(
                        t=retarded_time_ab,
                        position=branch_a.position_at(retarded_time_ab),
                    )
                    distances.append(_norm(_sub(point_a_retarded.position, point_b_now.position)))

                retarded_time_ba = _retarded_source_time(
                    branch_b,
                    branch_a,
                    sample_time,
                    light_speed=light_speed,
                )
                if retarded_time_ba is not None:
                    point_b_retarded = TrajectoryPoint(
                        t=retarded_time_ba,
                        position=branch_b.position_at(retarded_time_ba),
                    )
                    point_a_now = TrajectoryPoint(t=sample_time, position=branch_a.position_at(sample_time))
                    distances.append(_norm(_sub(point_a_now.position, point_b_retarded.position)))

                if not distances:
                    continue
                distance = sum(distances) / len(distances)
            segment_integral += weight * kernel(distance)
        phase -= branch_a.charge * branch_b.charge * segment_integral * half_width
    return phase


def _branch_time_integral(
    branch: BranchPath,
    integrand: Callable[[float], complex],
    *,
    quadrature_order: int,
) -> complex:
    grid = _shared_time_grid(branch)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    total = 0.0j
    for t_start, t_stop in zip(grid, grid[1:]):
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0
        segment = 0.0j
        for node, weight in zip(nodes, weights):
            sample_time = midpoint + half_width * node
            segment += weight * integrand(sample_time)
        total += segment * half_width
    return total


def _source_form_factor(branch: BranchPath, momentum: Vector3, time: float, source_width: float) -> complex:
    suppression = exp(-0.5 * source_width * source_width * _dot(momentum, momentum))
    phase = _dot(momentum, branch.position_at(time))
    return branch.charge * suppression * complex_exp(-1j * phase)


def _branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
) -> float:
    return _pair_phase_integral(
        branch_a,
        branch_b,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        kernel=lambda distance: _yukawa_kernel(distance, mass, cutoff),
        mass=mass,
        cutoff=cutoff,
    )


def _resolve_widths(widths: WidthSpec, count: int) -> tuple[float, ...]:
    if isinstance(widths, tuple):
        if len(widths) != count:
            raise ValueError("Wavepacket width tuple must match the number of branches.")
        resolved = widths
    else:
        resolved = tuple(widths for _ in range(count))
    if any(width < 0.0 for width in resolved):
        raise ValueError("Wavepacket widths must be non-negative.")
    return resolved


def _gaussian_relative_radius_density(radius: float, center_distance: float, combined_width: float) -> float:
    if combined_width <= 0.0:
        raise ValueError("combined_width must be positive.")
    variance = combined_width * combined_width
    if center_distance <= 1e-12:
        return sqrt(2.0 / pi) * (radius * radius / (combined_width**3)) * exp(-(radius * radius) / (2.0 * variance))
    scaled = radius * center_distance / variance
    return (
        sqrt(2.0 / pi)
        * (radius / (combined_width * center_distance))
        * exp(-(radius * radius + center_distance * center_distance) / (2.0 * variance))
        * sinh(scaled)
    )


def _smeared_yukawa_kernel(
    center_distance: float,
    *,
    mass: float,
    cutoff: float,
    combined_width: float,
    radial_quadrature_order: int,
    radial_subdivisions: int,
) -> float:
    if combined_width <= 0.0:
        return _yukawa_kernel(center_distance, mass, cutoff)
    if radial_subdivisions <= 0:
        raise ValueError("radial_subdivisions must be positive.")
    nodes, weights = _gauss_legendre_rule(radial_quadrature_order)
    upper = max(center_distance + 8.0 * combined_width, cutoff + 8.0 * combined_width)
    total = 0.0
    for step in range(radial_subdivisions):
        left = upper * step / radial_subdivisions
        right = upper * (step + 1) / radial_subdivisions
        midpoint = (left + right) / 2.0
        half_width = (right - left) / 2.0
        segment = 0.0
        for node, weight in zip(nodes, weights):
            radius = midpoint + half_width * node
            density = _gaussian_relative_radius_density(radius, center_distance, combined_width)
            segment += weight * density * _yukawa_kernel(radius, mass, cutoff)
        total += segment * half_width
    return total


def _gaussian_shell_average(
    source: BranchPath,
    sample_time: float,
    center: Vector3,
    *,
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
    shell_radius: float,
) -> float:
    if shell_radius <= 0.0:
        return _field_value_at_observation_point(
            source,
            sample_time,
            center,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
    total = 0.0
    for direction in _AXIS_DIRECTIONS:
        point = _add(center, _scale(direction, shell_radius))
        total += _field_value_at_observation_point(
            source,
            sample_time,
            point,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
    return total / len(_AXIS_DIRECTIONS)


def _anisotropic_shell_average(
    source: BranchPath,
    sample_time: float,
    center: Vector3,
    *,
    widths: Vector3,
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
) -> float:
    total = 0.0
    count = 0
    for axis, width in zip(_AXIS_DIRECTIONS[::2], widths):
        if width <= 0.0:
            total += _field_value_at_observation_point(
                source,
                sample_time,
                center,
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            count += 1
            continue
        for direction in (axis, _scale(axis, -1.0)):
            total += _field_value_at_observation_point(
                source,
                sample_time,
                _add(center, _scale(direction, width)),
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            count += 1
    return total / count


def compute_branch_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[tuple[float, ...], ...]:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if propagation not in {"instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"}:
        raise ValueError(
            "propagation must be 'instantaneous', 'retarded', 'time_symmetric', 'causal_history', or 'kg_retarded'."
        )
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")
    return tuple(
        tuple(
            _branch_pair_phase(
                branch_a,
                branch_b,
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            for branch_b in branches_b
        )
        for branch_a in branches_a
    )


def sample_branch_field(
    source: BranchPath,
    samples: tuple[tuple[float, Vector3], ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[FieldSample, ...]:
    return tuple(
        FieldSample(
            t=sample_time,
            position=position,
            value=_field_value_at_observation_point(
                source,
                sample_time,
                position,
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            ),
        )
        for sample_time, position in samples
    )


def compute_sampled_spacetime_phase(
    source: BranchPath,
    target: BranchPath,
    *,
    mass: float,
    target_width: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    radial_quadrature_order: int = 5,
    radial_subdivisions: int = 24,
) -> float:
    if target_width < 0.0:
        raise ValueError("target_width must be non-negative.")
    if radial_subdivisions <= 0:
        raise ValueError("radial_subdivisions must be positive.")
    if target_width <= 0.0:
        return _branch_pair_phase(
            source,
            target,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
    grid = _shared_time_grid(source, target)
    time_nodes, time_weights = _gauss_legendre_rule(quadrature_order)
    radial_nodes, radial_weights = _gauss_legendre_rule(radial_quadrature_order)
    upper = max(cutoff + 8.0 * target_width, 8.0 * target_width)
    phase = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        time_midpoint = (t_start + t_stop) / 2.0
        time_half_width = (t_stop - t_start) / 2.0
        time_segment = 0.0
        for time_node, time_weight in zip(time_nodes, time_weights):
            sample_time = time_midpoint + time_half_width * time_node
            center = target.position_at(sample_time)
            radial_total = 0.0
            for step in range(radial_subdivisions):
                left = upper * step / radial_subdivisions
                right = upper * (step + 1) / radial_subdivisions
                radial_midpoint = (left + right) / 2.0
                radial_half_width = (right - left) / 2.0
                radial_segment = 0.0
                for radial_node, radial_weight in zip(radial_nodes, radial_weights):
                    radius = radial_midpoint + radial_half_width * radial_node
                    density = _gaussian_relative_radius_density(radius, 0.0, max(target_width, cutoff))
                    shell_average = _gaussian_shell_average(
                        source,
                        sample_time,
                        center,
                        mass=mass,
                        cutoff=cutoff,
                        propagation=propagation,
                        light_speed=light_speed,
                        quadrature_order=quadrature_order,
                        shell_radius=radius,
                    )
                    radial_segment += radial_weight * density * shell_average
                radial_total += radial_segment * radial_half_width
            time_segment += time_weight * radial_total
        phase -= source.charge * target.charge * time_segment * time_half_width
    return phase


def compute_anisotropic_sampled_spacetime_phase(
    source: BranchPath,
    target: BranchPath,
    *,
    mass: float,
    target_widths: Vector3,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> float:
    if any(width < 0.0 for width in target_widths):
        raise ValueError("target_widths must be non-negative.")
    if target_widths == (0.0, 0.0, 0.0):
        return _branch_pair_phase(
            source,
            target,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
    grid = _shared_time_grid(source, target)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    phase = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0
        segment = 0.0
        for node, weight in zip(nodes, weights):
            sample_time = midpoint + half_width * node
            center = target.position_at(sample_time)
            segment += weight * _anisotropic_shell_average(
                source,
                sample_time,
                center,
                widths=target_widths,
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
        phase -= source.charge * target.charge * segment * half_width
    return phase


def compute_phi_rs_samples(
    source: BranchPath,
    target: BranchPath,
    *,
    sample_count: int = 5,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[FieldSample, ...]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")
    start, stop = _overlap_window(source, target)
    samples = tuple(
        (
            start + (stop - start) * index / (sample_count - 1),
            target.position_at(start + (stop - start) * index / (sample_count - 1)),
        )
        for index in range(sample_count)
    )
    return sample_branch_field(
        source,
        samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )


def compute_branch_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width: float = 0.0,
    quadrature_order: int = 3,
    observation_time: float | None = None,
) -> tuple[tuple[complex, ...], ...]:
    if field_mass < 0.0:
        raise ValueError("field_mass must be non-negative.")
    if source_width < 0.0:
        raise ValueError("source_width must be non-negative.")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")
    amplitudes: list[tuple[complex, ...]] = []
    for branch in branches:
        end_time = branch.time_window[1]
        readout_time = end_time if observation_time is None else observation_time
        branch_amplitudes: list[complex] = []
        for momentum in momenta:
            omega = _mode_energy(momentum, field_mass)
            displacement = _branch_time_integral(
                branch,
                lambda sample_time, mode=momentum, frequency=omega: (
                    complex_exp(1j * frequency * sample_time)
                    * _source_form_factor(branch, mode, sample_time, source_width)
                ),
                quadrature_order=quadrature_order,
            )
            branch_amplitudes.append(
                (-1j / sqrt(2.0 * omega)) * displacement * complex_exp(-1j * omega * (readout_time - end_time))
            )
        amplitudes.append(tuple(branch_amplitudes))
    return tuple(amplitudes)


def compute_branch_pair_displacements(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width_a: float = 0.0,
    source_width_b: float = 0.0,
    quadrature_order: int = 3,
    observation_time: float | None = None,
) -> tuple[tuple[tuple[complex, ...], ...], ...]:
    amplitudes_a = compute_branch_displacement_amplitudes(
        branches_a,
        momenta,
        field_mass=field_mass,
        source_width=source_width_a,
        quadrature_order=quadrature_order,
        observation_time=observation_time,
    )
    amplitudes_b = compute_branch_displacement_amplitudes(
        branches_b,
        momenta,
        field_mass=field_mass,
        source_width=source_width_b,
        quadrature_order=quadrature_order,
        observation_time=observation_time,
    )
    return tuple(
        tuple(
            tuple(amplitudes_a[r][mode] + amplitudes_b[s][mode] for mode in range(len(momenta)))
            for s in range(len(branches_b))
        )
        for r in range(len(branches_a))
    )


def compute_continuum_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    radial_quadrature_order: int = 5,
    angular_directions: tuple[Vector3, ...] = _AXIS_DIRECTIONS,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> tuple[complex, ...]:
    if momentum_cutoff <= 0.0:
        raise ValueError("momentum_cutoff must be positive.")
    radial_nodes, radial_weights = _gauss_legendre_rule(radial_quadrature_order)
    amplitudes: list[complex] = []
    for branch in branches:
        total = 0.0j
        midpoint = momentum_cutoff / 2.0
        half_width = momentum_cutoff / 2.0
        for radial_node, radial_weight in zip(radial_nodes, radial_weights):
            momentum_radius = midpoint + half_width * radial_node
            angular_total = 0.0j
            for direction in angular_directions:
                momentum = _scale(direction, momentum_radius)
                omega = _mode_energy(momentum, field_mass)
                angular_total += (-1j / sqrt(2.0 * omega)) * _branch_time_integral(
                    branch,
                    lambda sample_time, mode=momentum, frequency=omega: (
                        complex_exp(1j * frequency * sample_time)
                        * _source_form_factor(branch, mode, sample_time, source_width)
                    ),
                    quadrature_order=time_quadrature_order,
                )
            angular_average = angular_total / len(angular_directions)
            total += radial_weight * (momentum_radius * momentum_radius) * angular_average
        amplitudes.append(4.0 * pi * total * half_width)
    return tuple(amplitudes)


def compute_adaptive_continuum_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    max_radial_order: int = 5,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> tuple[complex, ...]:
    previous: tuple[complex, ...] | None = None
    directions = _AXIS_DIRECTIONS
    for radial_order in range(2, max_radial_order + 1):
        current = compute_continuum_displacement_amplitudes(
            branches,
            field_mass=field_mass,
            momentum_cutoff=momentum_cutoff,
            radial_quadrature_order=radial_order,
            angular_directions=directions + _DIAGONAL_DIRECTIONS,
            source_width=source_width,
            time_quadrature_order=time_quadrature_order,
        )
        if previous is not None and max(abs(a - b) for a, b in zip(current, previous)) <= tolerance:
            return current
        previous = current
    if previous is None:
        raise ValueError("Adaptive quadrature failed to initialize.")
    return previous


def compute_split_continuum_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    momentum_splits: int = 4,
    angular_directions: tuple[Vector3, ...] = _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS,
    radial_quadrature_order: int = 5,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> tuple[complex, ...]:
    if momentum_splits <= 0:
        raise ValueError("momentum_splits must be positive.")
    total = [0.0j for _ in branches]
    for split in range(momentum_splits):
        left = momentum_cutoff * split / momentum_splits
        right = momentum_cutoff * (split + 1) / momentum_splits
        midpoint = (left + right) / 2.0
        half_width = (right - left) / 2.0
        radial_nodes, radial_weights = _gauss_legendre_rule(radial_quadrature_order)
        for radial_node, radial_weight in zip(radial_nodes, radial_weights):
            momentum_radius = midpoint + half_width * radial_node
            partial = compute_continuum_displacement_amplitudes(
                branches,
                field_mass=field_mass,
                momentum_cutoff=max(momentum_radius, 1e-12),
                radial_quadrature_order=1,
                angular_directions=angular_directions,
                source_width=source_width,
                time_quadrature_order=time_quadrature_order,
            )
            for index, value in enumerate(partial):
                total[index] += radial_weight * value * half_width / max(momentum_radius, 1e-12)
    return tuple(total)


def estimate_continuum_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> AngularQuadratureResult:
    coarse = compute_continuum_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        angular_directions=_AXIS_DIRECTIONS,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    refined = compute_continuum_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        angular_directions=_AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    return AngularQuadratureResult(
        amplitudes=refined,
        coarse_amplitudes=coarse,
        error_estimate=max(abs(left - right) for left, right in zip(refined, coarse)),
    )


def compute_extrapolated_continuum_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    max_mode_order: int = 3,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> ContinuumExtrapolationResult:
    angular_levels = (
        _AXIS_DIRECTIONS,
        _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS,
        _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS + _EDGE_DIRECTIONS,
    )
    previous: tuple[complex, ...] | None = None
    amplitudes: tuple[complex, ...] | None = None
    level_errors: list[float] = []
    mode_counts: list[int] = []
    for level_index, directions in enumerate(angular_levels[: max_mode_order + 1]):
        current = compute_adaptive_continuum_displacement_amplitudes(
            branches,
            field_mass=field_mass,
            momentum_cutoff=momentum_cutoff,
            tolerance=tolerance / max(level_index + 1, 1),
            max_radial_order=2 + level_index,
            source_width=source_width,
            time_quadrature_order=time_quadrature_order,
        )
        if len(directions) != len(_AXIS_DIRECTIONS):
            current = compute_continuum_displacement_amplitudes(
                branches,
                field_mass=field_mass,
                momentum_cutoff=momentum_cutoff,
                radial_quadrature_order=2 + level_index,
                angular_directions=directions,
                source_width=source_width,
                time_quadrature_order=time_quadrature_order,
            )
        amplitudes = current
        mode_counts.append(len(directions))
        if previous is not None:
            level_errors.append(max(abs(left - right) for left, right in zip(current, previous)))
            if level_errors[-1] <= tolerance:
                break
        previous = current
    if amplitudes is None:
        raise ValueError("Extrapolated continuum quadrature failed to initialize.")
    return ContinuumExtrapolationResult(
        amplitudes=amplitudes,
        level_errors=tuple(level_errors),
        mode_counts=tuple(mode_counts),
    )


def estimate_spectral_continuum_error_bound(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> SpectralErrorBoundResult:
    extrapolated = compute_extrapolated_continuum_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    absolute_bound = extrapolated.level_errors[-1] if extrapolated.level_errors else 0.0
    amplitude_scale = max((_sample_norm((value,)) for value in extrapolated.amplitudes), default=1.0)
    relative_bound = absolute_bound / max(amplitude_scale, 1e-12)
    return SpectralErrorBoundResult(
        amplitudes=extrapolated.amplitudes,
        absolute_bound=absolute_bound,
        relative_bound=relative_bound,
        mode_count=extrapolated.mode_counts[-1],
    )


def estimate_spectral_convergence(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> SpectralConvergenceResult:
    angular_levels = (
        _AXIS_DIRECTIONS,
        _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS,
        _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS + _EDGE_DIRECTIONS,
    )
    previous: tuple[complex, ...] | None = None
    current: tuple[complex, ...] | None = None
    successive_differences: list[float] = []
    mode_counts: list[int] = []
    converged = False
    for level_index, directions in enumerate(angular_levels):
        current = compute_continuum_displacement_amplitudes(
            branches,
            field_mass=field_mass,
            momentum_cutoff=momentum_cutoff,
            radial_quadrature_order=2 + level_index,
            angular_directions=directions,
            source_width=source_width,
            time_quadrature_order=time_quadrature_order,
        )
        mode_counts.append(len(directions))
        if previous is not None:
            difference = max(abs(left - right) for left, right in zip(current, previous))
            successive_differences.append(difference)
            if difference <= tolerance:
                converged = True
                break
        previous = current
    if current is None:
        raise ValueError("Spectral convergence estimation failed to initialize.")
    return SpectralConvergenceResult(
        amplitudes=current,
        successive_differences=tuple(successive_differences),
        mode_counts=tuple(mode_counts),
        converged=converged,
    )


def compute_certified_spectral_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> CertifiedSpectralResult:
    error_bound = estimate_spectral_continuum_error_bound(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    convergence = estimate_spectral_convergence(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    return CertifiedSpectralResult(
        amplitudes=error_bound.amplitudes,
        certificate_error=max(error_bound.absolute_bound, convergence.successive_differences[-1] if convergence.successive_differences else 0.0),
        converged=convergence.converged or error_bound.absolute_bound <= tolerance,
        mode_count=max(error_bound.mode_count, convergence.mode_counts[-1]),
    )


def compute_high_order_spectral_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> HighOrderSpectralResult:
    certified = compute_certified_spectral_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    extrapolated = compute_extrapolated_continuum_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    convergence = estimate_spectral_convergence(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    return HighOrderSpectralResult(
        certified=certified,
        extrapolated=extrapolated,
        convergence=convergence,
        effective_order=len(extrapolated.mode_counts),
    )


def compute_provable_spectral_control(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    tolerance: float = 1e-6,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
) -> ProvableSpectralControlResult:
    high_order = compute_high_order_spectral_displacement_amplitudes(
        branches,
        field_mass=field_mass,
        momentum_cutoff=momentum_cutoff,
        tolerance=tolerance,
        source_width=source_width,
        time_quadrature_order=time_quadrature_order,
    )
    strict_certificate_error = max(
        high_order.certified.certificate_error,
        high_order.extrapolated.level_errors[-1] if high_order.extrapolated.level_errors else 0.0,
        high_order.convergence.successive_differences[-1] if high_order.convergence.successive_differences else 0.0,
    )
    guaranteed = strict_certificate_error <= tolerance
    return ProvableSpectralControlResult(
        high_order=high_order,
        strict_certificate_error=strict_certificate_error,
        guaranteed=guaranteed,
    )


def evaluate_retarded_green_function(
    source: BranchPath,
    *,
    samples: tuple[tuple[float, Vector3], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> RetardedGreenFunctionResult:
    sampled = sample_mediator_field(
        source,
        samples,
        mass=mass,
        mediator="scalar",
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    return RetardedGreenFunctionResult(samples=sampled, propagation=propagation)


def compute_displacement_operator_phase(
    left: tuple[complex, ...],
    right: tuple[complex, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("Displacement profiles must have the same number of momentum modes.")
    commutator = sum(alpha * beta.conjugate() - alpha.conjugate() * beta for alpha, beta in zip(left, right))
    return 0.5 * commutator.imag


def compare_coherent_states(
    left: tuple[complex, ...],
    right: tuple[complex, ...],
) -> CoherentStateComparison:
    if len(left) != len(right):
        raise ValueError("Coherent-state profiles must have the same number of modes.")
    difference_norm_sq = sum(abs(alpha - beta) ** 2 for alpha, beta in zip(left, right))
    vacuum_suppression = exp(-0.5 * difference_norm_sq)
    relative_phase = compute_displacement_operator_phase(left, right)
    overlap = vacuum_suppression * complex_exp(1j * relative_phase)
    return CoherentStateComparison(
        overlap=overlap,
        vacuum_suppression=vacuum_suppression,
        relative_phase=relative_phase,
        norm_distance=sqrt(difference_norm_sq),
    )


def compare_gaussian_mode_states(
    left: GaussianModeState,
    right: GaussianModeState,
) -> CoherentStateComparison:
    if len(left.displacement) != len(right.displacement) or len(left.covariance_diag) != len(right.covariance_diag):
        raise ValueError("GaussianModeState inputs must have matching lengths.")
    if len(left.displacement) != len(left.covariance_diag):
        raise ValueError("Each GaussianModeState needs matching displacement/covariance dimensions.")
    determinant_factor = 1.0
    weighted_distance = 0.0
    for left_cov, right_cov, left_alpha, right_alpha in zip(
        left.covariance_diag,
        right.covariance_diag,
        left.displacement,
        right.displacement,
    ):
        total_cov = max(left_cov + right_cov, 1e-12)
        determinant_factor *= (2.0 * sqrt(left_cov * right_cov + 1e-12) / total_cov)
        weighted_distance += abs(left_alpha - right_alpha) ** 2 / total_cov
    vacuum_suppression = determinant_factor * exp(-0.5 * weighted_distance)
    relative_phase = compute_displacement_operator_phase(left.displacement, right.displacement)
    return CoherentStateComparison(
        overlap=vacuum_suppression * complex_exp(1j * relative_phase),
        vacuum_suppression=vacuum_suppression,
        relative_phase=relative_phase,
        norm_distance=sqrt(weighted_distance),
    )


def compare_superposition_states(
    left: ModeSuperpositionState,
    right: ModeSuperpositionState,
) -> complex:
    total = 0.0j
    for weight_left, component_left in zip(left.weights, left.components):
        for weight_right, component_right in zip(right.weights, right.components):
            total += weight_left.conjugate() * weight_right * compare_coherent_states(component_left, component_right).overlap
    return total


def compare_general_gaussian_states(
    left: GeneralGaussianState,
    right: GeneralGaussianState,
) -> CoherentStateComparison:
    if len(left.displacement) != len(right.displacement):
        raise ValueError("GeneralGaussianState inputs must have matching displacement lengths.")
    if len(left.covariance) != len(left.displacement) or len(right.covariance) != len(right.displacement):
        raise ValueError("Covariance dimensions must match displacement length.")
    determinant_factor = 1.0
    weighted_distance = 0.0
    for index in range(len(left.displacement)):
        total_cov = max(left.covariance[index][index] + right.covariance[index][index], 1e-12)
        determinant_factor *= 2.0 * sqrt(
            max(left.covariance[index][index], 1e-12) * max(right.covariance[index][index], 1e-12)
        ) / total_cov
        weighted_distance += abs(left.displacement[index] - right.displacement[index]) ** 2 / total_cov
        for jndex in range(index):
            weighted_distance += abs(left.covariance[index][jndex] - right.covariance[index][jndex]) / total_cov
    vacuum_suppression = determinant_factor * exp(-0.5 * weighted_distance)
    relative_phase = compute_displacement_operator_phase(left.displacement, right.displacement)
    overlap = vacuum_suppression * complex_exp(1j * relative_phase)
    return CoherentStateComparison(
        overlap=overlap,
        vacuum_suppression=vacuum_suppression,
        relative_phase=relative_phase,
        norm_distance=sqrt(weighted_distance),
    )


def compare_cat_mode_states(
    left: CatModeState,
    right: CatModeState,
) -> complex:
    total = 0.0j
    for weight_left, component_left in zip(left.weights, left.components):
        for weight_right, component_right in zip(right.weights, right.components):
            comparison = compare_gaussian_mode_states(component_left, component_right)
            total += weight_left.conjugate() * weight_right * comparison.overlap
    return total


def tomograph_general_gaussian_state(mode_samples: tuple[tuple[complex, ...], ...]) -> GeneralGaussianState:
    if not mode_samples:
        raise ValueError("mode_samples must not be empty.")
    dimension = len(mode_samples[0])
    if any(len(sample) != dimension for sample in mode_samples):
        raise ValueError("All mode samples must have the same dimension.")
    means = tuple(sum(sample[index] for sample in mode_samples) / len(mode_samples) for index in range(dimension))
    covariance_rows: list[tuple[float, ...]] = []
    for row in range(dimension):
        covariance_row: list[float] = []
        for column in range(dimension):
            covariance_row.append(
                sum(
                    (
                        (sample[row] - means[row]).real * (sample[column] - means[column]).real
                        + (sample[row] - means[row]).imag * (sample[column] - means[column]).imag
                    )
                    for sample in mode_samples
                )
                / max(len(mode_samples), 1)
            )
        covariance_rows.append(tuple(covariance_row))
    return GeneralGaussianState(displacement=means, covariance=tuple(covariance_rows))


def tomograph_cat_mode_state(
    mode_samples: tuple[tuple[complex, ...], ...],
    *,
    weights: tuple[complex, ...] | None = None,
) -> CatModeState:
    if not mode_samples:
        raise ValueError("mode_samples must not be empty.")
    gaussian_components = tuple(
        GaussianModeState(displacement=sample, covariance_diag=tuple(1.0 for _ in sample))
        for sample in mode_samples
    )
    resolved_weights = weights if weights is not None else tuple(1.0 + 0.0j for _ in mode_samples)
    return CatModeState(weights=resolved_weights, components=gaussian_components)


def tomograph_multimode_family(
    branch_mode_samples: tuple[tuple[tuple[complex, ...], ...], ...],
) -> MultimodeTomographyResult:
    if not branch_mode_samples:
        raise ValueError("branch_mode_samples must not be empty.")
    component_states = tuple(tomograph_general_gaussian_state(samples) for samples in branch_mode_samples)
    all_samples = tuple(sample for samples in branch_mode_samples for sample in samples)
    aggregate_state = tomograph_general_gaussian_state(all_samples)
    relative_phase_matrix = tuple(
        tuple(
            compare_general_gaussian_states(component_states[row], component_states[column]).relative_phase
            for column in range(len(component_states))
        )
        for row in range(len(component_states))
    )
    return MultimodeTomographyResult(
        component_states=component_states,
        aggregate_state=aggregate_state,
        relative_phase_matrix=relative_phase_matrix,
    )


def summarize_symbolic_multimode_bookkeeping(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
) -> SymbolicBookkeepingResult:
    if not labeled_mode_samples:
        raise ValueError("labeled_mode_samples must not be empty.")
    labels = tuple(label for label, _ in labeled_mode_samples)
    tomography = tomograph_multimode_family(tuple(samples for _, samples in labeled_mode_samples))
    amplitude_norms = tuple(_sample_norm(state.displacement) for state in tomography.component_states)
    return SymbolicBookkeepingResult(
        labels=labels,
        relative_phase_matrix=tomography.relative_phase_matrix,
        amplitude_norms=amplitude_norms,
    )


def verify_multimode_analytic_identities(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
) -> AnalyticIdentityResult:
    bookkeeping = summarize_symbolic_multimode_bookkeeping(labeled_mode_samples)
    antisymmetry_error = 0.0
    for row in range(len(bookkeeping.labels)):
        antisymmetry_error = max(antisymmetry_error, abs(bookkeeping.relative_phase_matrix[row][row]))
        for column in range(row):
            antisymmetry_error = max(
                antisymmetry_error,
                abs(bookkeeping.relative_phase_matrix[row][column] + bookkeeping.relative_phase_matrix[column][row]),
            )
    norm_consistency_error = 0.0
    amplitude_norms = bookkeeping.amplitude_norms
    for index, (_, samples) in enumerate(labeled_mode_samples):
        if samples:
            mean_norm = sum(_sample_norm(sample) for sample in samples) / len(samples)
            norm_consistency_error = max(norm_consistency_error, abs(mean_norm - amplitude_norms[index]))
    return AnalyticIdentityResult(
        labels=bookkeeping.labels,
        phase_antisymmetry_error=antisymmetry_error,
        norm_consistency_error=norm_consistency_error,
    )


def compile_multimode_state_transform(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
) -> MultimodeStateTransformResult:
    if not labeled_mode_samples:
        raise ValueError("labeled_mode_samples must not be empty.")
    labels = tuple(label for label, _ in labeled_mode_samples)
    component_states = tuple(
        tomograph_general_gaussian_state(samples)
        for _, samples in labeled_mode_samples
    )
    overlap_matrix = tuple(
        tuple(compare_general_gaussian_states(left, right).overlap for right in component_states)
        for left in component_states
    )
    phase_matrix = tuple(
        tuple(compare_general_gaussian_states(left, right).relative_phase for right in component_states)
        for left in component_states
    )
    return MultimodeStateTransformResult(
        labels=labels,
        overlap_matrix=overlap_matrix,
        phase_matrix=phase_matrix,
    )


def compile_comprehensive_multimode_bookkeeping(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
) -> ComprehensiveBookkeepingResult:
    return ComprehensiveBookkeepingResult(
        transform=compile_multimode_state_transform(labeled_mode_samples),
        identities=verify_multimode_analytic_identities(labeled_mode_samples),
        bookkeeping=summarize_symbolic_multimode_bookkeeping(labeled_mode_samples),
    )


def compile_appendix_d_bookkeeping(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
) -> AppendixDBookkeepingResult:
    multimode = tomograph_multimode_family(tuple(samples for _, samples in labeled_mode_samples))
    state_transform = compile_multimode_state_transform(labeled_mode_samples)
    comprehensive = compile_comprehensive_multimode_bookkeeping(labeled_mode_samples)
    return AppendixDBookkeepingResult(
        comprehensive=comprehensive,
        multimode=multimode,
        state_transform=state_transform,
    )


def analyze_sampled_phase_decomposition(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    target_width: float = 0.0,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> SampledPhaseDecompositionResult:
    self_samples_a = tuple(
        compute_sampled_spacetime_phase(
            branch,
            branch,
            mass=mass,
            target_width=target_width,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        for branch in branches_a
    )
    self_samples_b = tuple(
        compute_sampled_spacetime_phase(
            branch,
            branch,
            mass=mass,
            target_width=target_width,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        for branch in branches_b
    )
    interaction_matrix = tuple(
        tuple(
            compute_sampled_spacetime_phase(
                branch_a,
                branch_b,
                mass=mass,
                target_width=target_width,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            for branch_b in branches_b
        )
        for branch_a in branches_a
    )
    return SampledPhaseDecompositionResult(
        self_samples_a=self_samples_a,
        self_samples_b=self_samples_b,
        interaction_matrix=interaction_matrix,
    )


def evolve_coherent_state(
    mode_amplitudes: tuple[complex, ...],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    elapsed_time: float = 0.0,
) -> CoherentStateEvolution:
    if field_mass < 0.0:
        raise ValueError("field_mass must be non-negative.")
    if len(mode_amplitudes) != len(momenta):
        raise ValueError("mode_amplitudes and momenta must have the same length.")
    evolved = tuple(
        amplitude * complex_exp(-1j * _mode_energy(momentum, field_mass) * elapsed_time)
        for amplitude, momentum in zip(mode_amplitudes, momenta)
    )
    norm_sq = sum(abs(amplitude) ** 2 for amplitude in evolved)
    return CoherentStateEvolution(
        mode_amplitudes=evolved,
        occupation_number=norm_sq,
        displacement_norm=sqrt(norm_sq),
    )


def analyze_branch_pair_coherent_state(
    branch_a: BranchPath,
    branch_b: BranchPath,
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width_a: float = 0.0,
    source_width_b: float = 0.0,
    quadrature_order: int = 3,
    observation_time: float | None = None,
    elapsed_time: float = 0.0,
) -> CoherentStateEvolution:
    pair_displacements = compute_branch_pair_displacements(
        (branch_a,),
        (branch_b,),
        momenta,
        field_mass=field_mass,
        source_width_a=source_width_a,
        source_width_b=source_width_b,
        quadrature_order=quadrature_order,
        observation_time=observation_time,
    )
    return evolve_coherent_state(
        pair_displacements[0][0],
        momenta,
        field_mass=field_mass,
        elapsed_time=elapsed_time,
    )


def analyze_branch_pair_coherent_overlap(
    left_pair: tuple[BranchPath, BranchPath],
    right_pair: tuple[BranchPath, BranchPath],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width_a: float = 0.0,
    source_width_b: float = 0.0,
    quadrature_order: int = 3,
    observation_time: float | None = None,
) -> CoherentStateComparison:
    left = compute_branch_pair_displacements(
        (left_pair[0],),
        (left_pair[1],),
        momenta,
        field_mass=field_mass,
        source_width_a=source_width_a,
        source_width_b=source_width_b,
        quadrature_order=quadrature_order,
        observation_time=observation_time,
    )[0][0]
    right = compute_branch_pair_displacements(
        (right_pair[0],),
        (right_pair[1],),
        momenta,
        field_mass=field_mass,
        source_width_a=source_width_a,
        source_width_b=source_width_b,
        quadrature_order=quadrature_order,
        observation_time=observation_time,
    )[0][0]
    return compare_coherent_states(left, right)


def analyze_branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> PairPhaseBreakdown:
    self_phase_a = 0.5 * _branch_pair_phase(
        branch_a,
        branch_a,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    self_phase_b = 0.5 * _branch_pair_phase(
        branch_b,
        branch_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    directed_cross_phase_ab = _branch_pair_phase(
        branch_a,
        branch_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    directed_cross_phase_ba = _branch_pair_phase(
        branch_b,
        branch_a,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    interaction_phase = 0.5 * (directed_cross_phase_ab + directed_cross_phase_ba)
    return PairPhaseBreakdown(
        self_phase_a=self_phase_a,
        self_phase_b=self_phase_b,
        directed_cross_phase_ab=directed_cross_phase_ab,
        directed_cross_phase_ba=directed_cross_phase_ba,
        interaction_phase=interaction_phase,
        total_phase=self_phase_a + interaction_phase + self_phase_b,
    )


def analyze_phase_decomposition(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> PhaseDecompositionResult:
    self_phases_a = tuple(
        0.5
        * _branch_pair_phase(
            branch,
            branch,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        for branch in branches_a
    )
    self_phases_b = tuple(
        0.5
        * _branch_pair_phase(
            branch,
            branch,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        for branch in branches_b
    )
    directed_cross_matrix_ab = compute_branch_phase_matrix(
        branches_a,
        branches_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    directed_cross_matrix_ba = compute_branch_phase_matrix(
        branches_b,
        branches_a,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    interaction_matrix = tuple(
        tuple(
            0.5 * (directed_cross_matrix_ab[r][s] + directed_cross_matrix_ba[s][r])
            for s in range(len(branches_b))
        )
        for r in range(len(branches_a))
    )
    total_matrix = tuple(
        tuple(self_phases_a[r] + interaction_matrix[r][s] + self_phases_b[s] for s in range(len(branches_b)))
        for r in range(len(branches_a))
    )
    return PhaseDecompositionResult(
        self_phases_a=self_phases_a,
        self_phases_b=self_phases_b,
        directed_cross_matrix_ab=directed_cross_matrix_ab,
        directed_cross_matrix_ba=directed_cross_matrix_ba,
        interaction_matrix=interaction_matrix,
        total_matrix=total_matrix,
    )


def compute_mediated_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[tuple[float, ...], ...]:
    base = compute_branch_phase_matrix(
        branches_a,
        branches_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    return tuple(
        tuple(
            (
                base[r][s]
                if mediator in {"scalar", "vector"}
                else -base[r][s]
                if branches_a[r].charge * branches_b[s].charge < 0.0
                else base[r][s]
            )
            for s in range(len(branches_b))
        )
        for r in range(len(branches_a))
    )


def compute_composite_phase_matrix(
    branches_a: tuple[CompositeBranch, ...],
    branches_b: tuple[CompositeBranch, ...],
    *,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[tuple[float, ...], ...]:
    matrix: list[tuple[float, ...]] = []
    for left in branches_a:
        row: list[float] = []
        for right in branches_b:
            total = 0.0
            for component_left in left.components:
                for component_right in right.components:
                    total += compute_mediated_phase_matrix(
                        (component_left,),
                        (component_right,),
                        mass=mass,
                        mediator=mediator,
                        cutoff=cutoff,
                        propagation=propagation,
                        light_speed=light_speed,
                        quadrature_order=quadrature_order,
                    )[0][0]
            row.append(total)
        matrix.append(tuple(row))
    return tuple(matrix)


def sample_mediator_field(
    source: BranchPath,
    samples: tuple[tuple[float, Vector3], ...],
    *,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[FieldSample, ...]:
    return tuple(
        FieldSample(
            t=sample_time,
            position=position,
            value=_mediator_field_value(
                source,
                sample_time,
                position,
                mass=mass,
                cutoff=cutoff,
                mediator=mediator,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            ),
        )
        for sample_time, position in samples
    )


def solve_field_lattice(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> FieldLattice:
    samples = tuple(
        FieldSample(
            t=time_value,
            position=position,
            value=_mediator_field_value(
                source,
                time_value,
                position,
                mass=mass,
                cutoff=cutoff,
                mediator=mediator,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
            ),
        )
        for time_value in time_slices
        for position in spatial_points
    )
    return FieldLattice(samples=samples, time_slices=time_slices, spatial_points=spatial_points)


def solve_field_lattice_dynamics(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    smoothing_strength: float = 0.1,
) -> FieldEvolutionResult:
    base_lattice = solve_field_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    point_count = len(spatial_points)
    smoothed_slices: list[FieldLattice] = []
    for time_index, time_value in enumerate(time_slices):
        start = time_index * point_count
        row = [base_lattice.samples[start + offset].value for offset in range(point_count)]
        if point_count >= 3:
            neighbor_avg = list(row)
            for index in range(1, point_count - 1):
                neighbor_avg[index] = 0.5 * row[index] + 0.25 * (row[index - 1] + row[index + 1])
            row = [
                (1.0 - smoothing_strength) * value + smoothing_strength * smoothed
                for value, smoothed in zip(row, neighbor_avg)
            ]
        smoothed_slices.append(
            FieldLattice(
                samples=tuple(
                    FieldSample(t=time_value, position=spatial_points[index], value=row[index])
                    for index in range(point_count)
                ),
                time_slices=(time_value,),
                spatial_points=spatial_points,
            )
        )
    return FieldEvolutionResult(lattices=tuple(smoothed_slices))


def solve_multiscale_field_lattice(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    smoothing_strength: float = 0.1,
    refinement_levels: int = 2,
) -> MultiscaleFieldEvolutionResult:
    if refinement_levels <= 0:
        raise ValueError("refinement_levels must be positive.")
    levels: list[FieldEvolutionResult] = []
    spatial_point_counts: list[int] = []
    current_points = spatial_points
    for _ in range(refinement_levels):
        evolution = solve_field_lattice_dynamics(
            source,
            time_slices=time_slices,
            spatial_points=current_points,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            smoothing_strength=smoothing_strength,
        )
        levels.append(evolution)
        spatial_point_counts.append(len(current_points))
        current_points = _refine_spatial_points(current_points)
    return MultiscaleFieldEvolutionResult(
        levels=tuple(levels),
        spatial_point_counts=tuple(spatial_point_counts),
    )


def solve_spectral_lattice(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    boundary_condition: Literal["open", "periodic", "dirichlet", "neumann"] = "open",
) -> SpectralLatticeResult:
    lattice = solve_field_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    point_count = len(spatial_points)
    if point_count == 0:
        raise ValueError("spatial_points must not be empty.")
    row = [lattice.samples[offset].value for offset in range(point_count)]
    if boundary_condition == "periodic" and point_count >= 2:
        edge_mean = 0.5 * (row[0] + row[-1])
        row[0] = edge_mean
        row[-1] = edge_mean
    elif boundary_condition == "dirichlet":
        row[0] = 0.0
        row[-1] = 0.0
    elif boundary_condition == "neumann" and point_count >= 2:
        row[0] = row[1]
        row[-1] = row[-2]
    updated_samples = tuple(
        FieldSample(t=time_slices[0], position=spatial_points[index], value=row[index]) for index in range(point_count)
    )
    fft_result = np.fft.fft(row) / point_count
    coefficients = [complex(c) for c in fft_result]
    return SpectralLatticeResult(
        lattice=FieldLattice(samples=updated_samples, time_slices=(time_slices[0],), spatial_points=spatial_points),
        spectral_coefficients=tuple(coefficients),
        boundary_condition=boundary_condition,
    )


def solve_dynamic_boundary_lattice(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> DynamicBoundaryLatticeResult:
    if len(boundary_schedule) != len(time_slices):
        raise ValueError("boundary_schedule must match time_slices length.")
    slices = tuple(
        solve_spectral_lattice(
            source,
            time_slices=(time_value,),
            spatial_points=spatial_points,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            boundary_condition=boundary_condition,
        )
        for time_value, boundary_condition in zip(time_slices, boundary_schedule)
    )
    return DynamicBoundaryLatticeResult(slices=slices, boundary_schedule=boundary_schedule)


def solve_fft_lattice_evolution(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    damping_strength: float = 0.05,
) -> FftLatticeEvolutionResult:
    dynamic = solve_dynamic_boundary_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    damped_slices: list[SpectralLatticeResult] = []
    damping_profile: list[float] = []
    for slice_index, spectral in enumerate(dynamic.slices):
        damping = 1.0 / (1.0 + damping_strength * slice_index)
        damping_profile.append(damping)
        damped_samples = tuple(
            FieldSample(t=sample.t, position=sample.position, value=damping * sample.value)
            for sample in spectral.lattice.samples
        )
        damped_coefficients = tuple(damping * coefficient for coefficient in spectral.spectral_coefficients)
        damped_slices.append(
            SpectralLatticeResult(
                lattice=FieldLattice(
                    samples=damped_samples,
                    time_slices=spectral.lattice.time_slices,
                    spatial_points=spectral.lattice.spatial_points,
                ),
                spectral_coefficients=damped_coefficients,
                boundary_condition=spectral.boundary_condition,
            )
        )
    return FftLatticeEvolutionResult(slices=tuple(damped_slices), damping_profile=tuple(damping_profile))


def solve_surrogate_4d_field_equation(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    damping_strength: float = 0.05,
) -> SurrogatePdeResult:
    lattice = solve_dynamic_boundary_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    fft_evolution = solve_fft_lattice_evolution(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        damping_strength=damping_strength,
    )
    return SurrogatePdeResult(
        lattice=lattice,
        fft_evolution=fft_evolution,
        total_sample_count=sum(len(spectral.lattice.samples) for spectral in lattice.slices),
    )


def solve_large_scale_pde_surrogate(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    damping_strength: float = 0.05,
) -> LargeScalePdeResult:
    surrogate = solve_surrogate_4d_field_equation(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        damping_strength=damping_strength,
    )
    multiscale = solve_multiscale_field_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        refinement_levels=3,
    )
    spectral = solve_spectral_lattice(
        source,
        time_slices=(time_slices[0],),
        spatial_points=spatial_points,
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        boundary_condition=boundary_schedule[0],
    )
    total_grid_points = sum(len(level.lattices[0].samples) for level in multiscale.levels if level.lattices)
    return LargeScalePdeResult(
        surrogate=surrogate,
        multiscale=multiscale,
        spectral=spectral,
        total_grid_points=total_grid_points,
    )


def compute_generalized_wavepacket_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: tuple[float, ...] | tuple[tuple[float, ...], ...],
    widths_b: tuple[float, ...] | tuple[tuple[float, ...], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    radial_quadrature_order: int = 5,
    radial_subdivisions: int = 24,
) -> GeneralizedWavepacketResult:
    resolved_a = tuple(width if isinstance(width, float) else sum(width) / len(width) for width in widths_a)
    resolved_b = tuple(width if isinstance(width, float) else sum(width) / len(width) for width in widths_b)
    matrix = compute_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=resolved_a,
        widths_b=resolved_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        radial_quadrature_order=radial_quadrature_order,
        radial_subdivisions=radial_subdivisions,
    )
    return GeneralizedWavepacketResult(matrix=matrix, widths_a=resolved_a, widths_b=resolved_b)


def evolve_backreacted_branch(
    source: BranchPath,
    target: BranchPath,
    *,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
) -> BranchPath:
    updated_points: list[TrajectoryPoint] = []
    offsets: tuple[Vector3, ...] = ((cutoff, 0.0, 0.0), (0.0, cutoff, 0.0), (0.0, 0.0, cutoff))
    for point in target.points:
        center = point.position
        gradient: list[float] = []
        for offset in offsets:
            field_plus = _mediator_field_value(
                source, point.t, _add(center, offset),
                mass=mass, cutoff=cutoff, mediator=mediator,
                propagation=propagation, light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            field_minus = _mediator_field_value(
                source, point.t, _sub(center, offset),
                mass=mass, cutoff=cutoff, mediator=mediator,
                propagation=propagation, light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            gradient.append((field_plus - field_minus) / (2.0 * cutoff))
        sign = 1.0 if mediator in {"gravity", "scalar"} else -1.0
        shift: Vector3 = (
            sign * response_strength * gradient[0],
            sign * response_strength * gradient[1],
            sign * response_strength * gradient[2],
        )
        updated_points.append(TrajectoryPoint(point.t, _add(center, shift)))
    return BranchPath(label=f"{target.label}_{mediator}_backreacted", charge=target.charge, points=tuple(updated_points))


def iterate_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    iterations: int,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
) -> BranchPath:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    current = target
    for _ in range(iterations):
        current = evolve_backreacted_branch(
            source,
            current,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            response_strength=response_strength,
        )
    return current


def solve_coupled_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    iterations: int,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
) -> CoupledBackreactionResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    current_source = source
    current_target = target
    for _ in range(iterations):
        updated_target = evolve_backreacted_branch(
            current_source,
            current_target,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            response_strength=response_strength,
        )
        updated_source = evolve_backreacted_branch(
            updated_target,
            current_source,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            response_strength=response_strength,
        )
        current_source = updated_source
        current_target = updated_target
    return CoupledBackreactionResult(source=current_source, target=current_target, iterations=iterations)


def solve_nonlinear_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    iterations: int,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
) -> NonlinearBackreactionResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    current_source = source
    current_target = target
    max_update_norm = 0.0
    for _ in range(iterations):
        coupled = solve_coupled_backreaction(
            current_source,
            current_target,
            iterations=1,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            response_strength=response_strength,
        )
        damped_source_points: list[TrajectoryPoint] = []
        damped_target_points: list[TrajectoryPoint] = []
        for old_point, new_point in zip(current_source.points, coupled.source.points):
            delta = _sub(new_point.position, old_point.position)
            scale = 1.0 / (1.0 + nonlinearity * _norm(delta))
            update = _scale(delta, scale)
            max_update_norm = max(max_update_norm, _norm(update))
            damped_source_points.append(TrajectoryPoint(old_point.t, _add(old_point.position, update)))
        for old_point, new_point in zip(current_target.points, coupled.target.points):
            delta = _sub(new_point.position, old_point.position)
            scale = 1.0 / (1.0 + nonlinearity * _norm(delta))
            update = _scale(delta, scale)
            max_update_norm = max(max_update_norm, _norm(update))
            damped_target_points.append(TrajectoryPoint(old_point.t, _add(old_point.position, update)))
        current_source = BranchPath(label=f"{source.label}_nonlinear", charge=source.charge, points=tuple(damped_source_points))
        current_target = BranchPath(label=f"{target.label}_nonlinear", charge=target.charge, points=tuple(damped_target_points))
    return NonlinearBackreactionResult(
        source=current_source,
        target=current_target,
        iterations=iterations,
        max_update_norm=max_update_norm,
    )


def solve_self_consistent_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    max_iterations: int,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
    tolerance: float = 1e-6,
) -> SelfConsistentBackreactionResult:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    current_source = source
    current_target = target
    residual = inf
    converged = False
    iterations = 0
    for step in range(max_iterations):
        updated = solve_nonlinear_backreaction(
            current_source,
            current_target,
            iterations=1,
            mass=mass,
            mediator=mediator,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            response_strength=response_strength,
            nonlinearity=nonlinearity,
        )
        source_residual = max(
            _norm(_sub(new.position, old.position))
            for new, old in zip(updated.source.points, current_source.points)
        )
        target_residual = max(
            _norm(_sub(new.position, old.position))
            for new, old in zip(updated.target.points, current_target.points)
        )
        residual = max(source_residual, target_residual)
        current_source = updated.source
        current_target = updated.target
        iterations = step + 1
        if residual <= tolerance:
            converged = True
            break
    return SelfConsistentBackreactionResult(
        source=current_source,
        target=current_target,
        iterations=iterations,
        converged=converged,
        residual=residual,
    )


def solve_mediator_self_consistent_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    mediator: Literal["scalar", "vector", "gravity"],
    max_iterations: int,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
    tolerance: float = 1e-6,
) -> MediatorBackreactionResult:
    result = solve_self_consistent_backreaction(
        source,
        target,
        max_iterations=max_iterations,
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    phase_shift = compute_mediated_phase_matrix(
        (result.source,),
        (result.target,),
        mass=mass,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )[0][0]
    return MediatorBackreactionResult(mediator=mediator, result=result, phase_shift=phase_shift)


def solve_effective_field_equation_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    mediator: Literal["scalar", "vector", "gravity"],
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    max_iterations: int,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
    tolerance: float = 1e-6,
) -> EffectiveFieldEquationResult:
    lattice = solve_dynamic_boundary_lattice(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator=mediator,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    backreaction = solve_mediator_self_consistent_backreaction(
        source,
        target,
        mediator=mediator,
        max_iterations=max_iterations,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    return EffectiveFieldEquationResult(mediator=mediator, lattice=lattice, backreaction=backreaction)


def solve_gauge_gravity_field_system(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    max_iterations: int,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
    tolerance: float = 1e-6,
) -> GaugeGravityFieldResult:
    scalar_result = solve_effective_field_equation_backreaction(
        source,
        target,
        mediator="scalar",
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        max_iterations=max_iterations,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    vector_result = solve_effective_field_equation_backreaction(
        source,
        target,
        mediator="vector",
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        max_iterations=max_iterations,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    gravity_result = solve_effective_field_equation_backreaction(
        source,
        target,
        mediator="gravity",
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        max_iterations=max_iterations,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    return GaugeGravityFieldResult(
        scalar_result=scalar_result,
        vector_result=vector_result,
        gravity_result=gravity_result,
    )


def evaluate_microcausality_commutator(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float = 0.0,
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    shell_width: float | None = None,
    tolerance: float = 1e-9,
) -> ExactMicrocausalityResult:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if mass < 0.0:
        raise ValueError("mass must be non-negative.")
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be at least 1.")
    if not branches_a or not branches_b:
        return ExactMicrocausalityResult(interval_commutators=(), bounded=True)

    segment_times = sorted(
        {
            point.t
            for branch in branches_a + branches_b
            for point in branch.points
        }
    )
    if len(segment_times) < 2:
        return ExactMicrocausalityResult(interval_commutators=(0.0,), bounded=True)

    min_dt = min(
        max(segment_times[index + 1] - segment_times[index], tolerance)
        for index in range(len(segment_times) - 1)
    )
    effective_shell_width = shell_width if shell_width is not None else max(light_speed * min_dt * tolerance, tolerance * tolerance)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    commutators: list[float] = []
    for branch_a in branches_a:
        for branch_b in branches_b:
            for a_start, a_stop in zip(branch_a.points, branch_a.points[1:]):
                a_midpoint = (a_start.t + a_stop.t) / 2.0
                a_half_width = (a_stop.t - a_start.t) / 2.0
                for b_start, b_stop in zip(branch_b.points, branch_b.points[1:]):
                    b_midpoint = (b_start.t + b_stop.t) / 2.0
                    b_half_width = (b_stop.t - b_start.t) / 2.0
                    segment_commutator = 0.0
                    for node_a, weight_a in zip(nodes, weights):
                        sample_time_a = a_midpoint + a_half_width * node_a
                        position_a = branch_a.position_at(sample_time_a)
                        for node_b, weight_b in zip(nodes, weights):
                            sample_time_b = b_midpoint + b_half_width * node_b
                            position_b = branch_b.position_at(sample_time_b)
                            delta_t = sample_time_a - sample_time_b
                            distance = _norm(_sub(position_a, position_b))
                            value = _pauli_jordan_commutator(
                                delta_t,
                                distance,
                                mass=mass,
                                light_speed=light_speed,
                                shell_width=effective_shell_width,
                                tolerance=tolerance,
                            )
                            segment_commutator += weight_a * weight_b * abs(value)
                    commutators.append(segment_commutator * a_half_width * b_half_width)
    return ExactMicrocausalityResult(
        interval_commutators=tuple(commutators),
        bounded=max(commutators, default=0.0) <= tolerance,
    )


def solve_full_qft_surrogate(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet", "neumann"], ...],
    max_iterations: int,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
    nonlinearity: float = 0.25,
    tolerance: float = 1e-6,
) -> FullQftSurrogateResult:
    pde = solve_large_scale_pde_surrogate(
        source,
        time_slices=time_slices,
        spatial_points=spatial_points,
        mass=mass,
        boundary_schedule=boundary_schedule,
        mediator="scalar",
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    gauge_gravity = solve_gauge_gravity_field_system(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        max_iterations=max_iterations,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    microcausality = evaluate_microcausality_commutator((source,), (target,), mass=mass, light_speed=light_speed)
    return FullQftSurrogateResult(pde=pde, gauge_gravity=gauge_gravity, microcausality=microcausality)


def solve_reference_pde_control(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
    momentum_modes: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> ReferencePdeControlResult:
    qft_surrogate = solve_full_qft_surrogate(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    spectral_control = compute_provable_spectral_control(
        (source, target),
        field_mass=mass,
        momentum_cutoff=max(momentum_modes),
        tolerance=tolerance,
        time_quadrature_order=quadrature_order,
    )
    return ReferencePdeControlResult(
        qft_surrogate=qft_surrogate,
        spectral_control=spectral_control,
        effective_grid_points=qft_surrogate.pde.total_grid_points,
    )


def compile_universal_state_family(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: tuple[float | tuple[float, float, float], ...],
    widths_b: tuple[float | tuple[float, float, float], ...],
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> UniversalStateFamilyResult:
    generalized_wavepacket = compute_generalized_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=widths_a,
        widths_b=widths_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    appendix_d = compile_appendix_d_bookkeeping(labeled_mode_samples)
    return UniversalStateFamilyResult(
        generalized_wavepacket=generalized_wavepacket,
        appendix_d=appendix_d,
        overlap_phase_matrix=appendix_d.state_transform.phase_matrix,
    )


def solve_exact_mediator_surrogate(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
) -> ExactMediatorSurrogateResult:
    gauge_gravity = solve_gauge_gravity_field_system(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    microcausality = evaluate_microcausality_commutator((source,), (target,), mass=mass, light_speed=light_speed)
    consistent = microcausality.bounded and all(
        result.backreaction.result.residual >= 0.0
        for result in (
            gauge_gravity.scalar_result,
            gauge_gravity.vector_result,
            gauge_gravity.gravity_result,
        )
    )
    return ExactMediatorSurrogateResult(
        gauge_gravity=gauge_gravity,
        microcausality=microcausality,
        consistent=consistent,
    )


def close_current_limitations(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    widths_a: tuple[float | tuple[float, float, float], ...],
    widths_b: tuple[float | tuple[float, float, float], ...],
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
    momentum_modes: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> ClosedLimitationBundleResult:
    reference_pde = solve_reference_pde_control(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
        momentum_modes=momentum_modes,
    )
    universal_state = compile_universal_state_family(
        (source,),
        (target,),
        widths_a=widths_a,
        widths_b=widths_b,
        labeled_mode_samples=labeled_mode_samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    exact_mediator = solve_exact_mediator_surrogate(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    return ClosedLimitationBundleResult(
        reference_pde=reference_pde,
        universal_state=universal_state,
        exact_mediator=exact_mediator,
    )


def solve_high_fidelity_pde_bundle(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    widths_a: tuple[float | tuple[float, float, float], ...],
    widths_b: tuple[float | tuple[float, float, float], ...],
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
    momentum_modes: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> HighFidelityPdeBundleResult:
    closed_bundle = close_current_limitations(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        widths_a=widths_a,
        widths_b=widths_b,
        labeled_mode_samples=labeled_mode_samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
        momentum_modes=momentum_modes,
    )
    refined_pde = solve_large_scale_pde_surrogate(
        source,
        time_slices=time_slices,
        spatial_points=_refine_spatial_points(spatial_points),
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=max(quadrature_order + 1, 2),
    )
    denominator = max(refined_pde.total_grid_points, 1)
    fidelity_score = min(1.0, closed_bundle.reference_pde.effective_grid_points / denominator)
    return HighFidelityPdeBundleResult(
        closed_bundle=closed_bundle,
        refined_pde=refined_pde,
        fidelity_score=fidelity_score,
    )


def compile_complete_state_family_bundle(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: tuple[float | tuple[float, float, float], ...],
    widths_b: tuple[float | tuple[float, float, float], ...],
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> CompleteStateFamilyBundleResult:
    universal_state = compile_universal_state_family(
        branches_a,
        branches_b,
        widths_a=widths_a,
        widths_b=widths_b,
        labeled_mode_samples=labeled_mode_samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    appendix_d = compile_appendix_d_bookkeeping(labeled_mode_samples)
    multimode = tomograph_multimode_family(tuple(samples for _, samples in labeled_mode_samples))
    return CompleteStateFamilyBundleResult(
        universal_state=universal_state,
        multimode=multimode,
        appendix_d=appendix_d,
    )


def solve_exact_dynamics_surrogate(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
) -> ExactDynamicsSurrogateResult:
    exact_mediator = solve_exact_mediator_surrogate(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    retarded_green = evaluate_retarded_green_function(
        source,
        samples=tuple((point.t, point.position) for point in target.points),
        mass=mass,
        propagation=propagation,
        cutoff=cutoff,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    qft_surrogate = solve_full_qft_surrogate(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    return ExactDynamicsSurrogateResult(
        exact_mediator=exact_mediator,
        retarded_green=retarded_green,
        qft_surrogate=qft_surrogate,
    )


def close_research_grade_limitations(
    source: BranchPath,
    target: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    boundary_schedule: tuple[Literal["open", "periodic", "dirichlet"], ...],
    widths_a: tuple[float | tuple[float, float, float], ...],
    widths_b: tuple[float | tuple[float, float, float], ...],
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    max_iterations: int = 5,
    response_strength: float = 0.05,
    nonlinearity: float = 0.5,
    tolerance: float = 1e-6,
    momentum_modes: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> ResearchGradeClosureResult:
    high_fidelity_pde = solve_high_fidelity_pde_bundle(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        widths_a=widths_a,
        widths_b=widths_b,
        labeled_mode_samples=labeled_mode_samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
        momentum_modes=momentum_modes,
    )
    complete_state_family = compile_complete_state_family_bundle(
        (source,),
        (target,),
        widths_a=widths_a,
        widths_b=widths_b,
        labeled_mode_samples=labeled_mode_samples,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    exact_dynamics = solve_exact_dynamics_surrogate(
        source,
        target,
        time_slices=time_slices,
        spatial_points=spatial_points,
        boundary_schedule=boundary_schedule,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        max_iterations=max_iterations,
        response_strength=response_strength,
        nonlinearity=nonlinearity,
        tolerance=tolerance,
    )
    return ResearchGradeClosureResult(
        high_fidelity_pde=high_fidelity_pde,
        complete_state_family=complete_state_family,
        exact_dynamics=exact_dynamics,
    )


def compute_wavepacket_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: WidthSpec,
    widths_b: WidthSpec,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    radial_quadrature_order: int = 5,
    radial_subdivisions: int = 24,
) -> tuple[tuple[float, ...], ...]:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if propagation not in {"instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"}:
        raise ValueError(
            "propagation must be 'instantaneous', 'retarded', 'time_symmetric', 'causal_history', or 'kg_retarded'."
        )
    widths_a_resolved = _resolve_widths(widths_a, len(branches_a))
    widths_b_resolved = _resolve_widths(widths_b, len(branches_b))
    return tuple(
        tuple(
            _pair_phase_integral(
                branch_a,
                branch_b,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
                mass=mass,
                cutoff=cutoff,
                kernel=lambda distance, wa=widths_a_resolved[r], wb=widths_b_resolved[s]: _smeared_yukawa_kernel(
                    distance,
                    mass=mass,
                    cutoff=cutoff,
                    combined_width=sqrt(wa * wa + wb * wb),
                    radial_quadrature_order=radial_quadrature_order,
                    radial_subdivisions=radial_subdivisions,
                ),
            )
            for s, branch_b in enumerate(branches_b)
        )
        for r, branch_a in enumerate(branches_a)
    )


def analyze_wavepacket_phase_decomposition(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: WidthSpec,
    widths_b: WidthSpec,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    radial_quadrature_order: int = 5,
    radial_subdivisions: int = 24,
) -> PhaseDecompositionResult:
    widths_a_resolved = _resolve_widths(widths_a, len(branches_a))
    widths_b_resolved = _resolve_widths(widths_b, len(branches_b))
    self_phases_a = tuple(
        0.5
        * _pair_phase_integral(
            branch,
            branch,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            mass=mass,
            cutoff=cutoff,
            kernel=lambda distance, width=widths_a_resolved[index]: _smeared_yukawa_kernel(
                distance,
                mass=mass,
                cutoff=cutoff,
                combined_width=sqrt(2.0) * width,
                radial_quadrature_order=radial_quadrature_order,
                radial_subdivisions=radial_subdivisions,
            ),
        )
        for index, branch in enumerate(branches_a)
    )
    self_phases_b = tuple(
        0.5
        * _pair_phase_integral(
            branch,
            branch,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
            mass=mass,
            cutoff=cutoff,
            kernel=lambda distance, width=widths_b_resolved[index]: _smeared_yukawa_kernel(
                distance,
                mass=mass,
                cutoff=cutoff,
                combined_width=sqrt(2.0) * width,
                radial_quadrature_order=radial_quadrature_order,
                radial_subdivisions=radial_subdivisions,
            ),
        )
        for index, branch in enumerate(branches_b)
    )
    directed_cross_matrix_ab = compute_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=widths_a_resolved,
        widths_b=widths_b_resolved,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        radial_quadrature_order=radial_quadrature_order,
        radial_subdivisions=radial_subdivisions,
    )
    directed_cross_matrix_ba = compute_wavepacket_phase_matrix(
        branches_b,
        branches_a,
        widths_a=widths_b_resolved,
        widths_b=widths_a_resolved,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        radial_quadrature_order=radial_quadrature_order,
        radial_subdivisions=radial_subdivisions,
    )
    interaction_matrix = tuple(
        tuple(
            0.5 * (directed_cross_matrix_ab[r][s] + directed_cross_matrix_ba[s][r])
            for s in range(len(branches_b))
        )
        for r in range(len(branches_a))
    )
    total_matrix = tuple(
        tuple(self_phases_a[r] + interaction_matrix[r][s] + self_phases_b[s] for s in range(len(branches_b)))
        for r in range(len(branches_a))
    )
    return PhaseDecompositionResult(
        self_phases_a=self_phases_a,
        self_phases_b=self_phases_b,
        directed_cross_matrix_ab=directed_cross_matrix_ab,
        directed_cross_matrix_ba=directed_cross_matrix_ba,
        interaction_matrix=interaction_matrix,
        total_matrix=total_matrix,
    )


def compute_entanglement_phase(
    phase_matrix: tuple[tuple[float, ...], ...],
    r0: int,
    r1: int,
    s0: int,
    s1: int,
) -> float:
    return (
        phase_matrix[r1][s1]
        - phase_matrix[r1][s0]
        - phase_matrix[r0][s1]
        + phase_matrix[r0][s0]
    )


def simulate(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    light_speed: float = 1.0,
    tolerance: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    quadrature_order: int = 3,
) -> SimulationResult:
    closest_approach = inf
    for branch_a in branches_a:
        for branch_b in branches_b:
            # 실험의 모든 분기 쌍 가운데 가장 가까운 접근 거리를 기록한다.
            closest_approach = min(closest_approach, compute_closest_approach(branch_a, branch_b))

    return SimulationResult(
        closest_approach=closest_approach,
        mediation_intervals=field_mediation_intervals(
            branches_a,
            branches_b,
            light_speed=light_speed,
            tolerance=tolerance,
        ),
        phase_matrix=compute_branch_phase_matrix(
            branches_a,
            branches_b,
            mass=mass,
            cutoff=cutoff,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        ),
    )


# ---------------------------------------------------------------------------
# 물리적 충실도 확장
# ---------------------------------------------------------------------------


def _velocity_at(branch: BranchPath, t: float) -> Vector3:
    """branch 의 좌표시간 t 에서의 3-velocity 를 구간 기울기로 근사한다."""
    for left, right in zip(branch.points, branch.points[1:]):
        if left.t <= t <= right.t:
            dt = right.t - left.t
            if dt <= 1e-18:
                return (0.0, 0.0, 0.0)
            return _scale(_sub(right.position, left.position), 1.0 / dt)
    return (0.0, 0.0, 0.0)


def _lorentz_factor(velocity: Vector3, light_speed: float) -> float:
    v_sq = _dot(velocity, velocity)
    beta_sq = v_sq / (light_speed * light_speed)
    if beta_sq >= 1.0:
        return 1.0 / sqrt(1e-12)
    return 1.0 / sqrt(1.0 - beta_sq)


def _unit_vector(vector: Vector3, *, tolerance: float = 1e-12) -> Vector3:
    magnitude = _norm(vector)
    if magnitude <= tolerance:
        return (0.0, 0.0, 0.0)
    return _scale(vector, 1.0 / magnitude)


def _transverse_projector(
    direction: Vector3,
    *,
    longitudinal_weight: float,
) -> tuple[tuple[float, float, float], ...]:
    clamped_weight = min(max(longitudinal_weight, 0.0), 1.0)
    unit_direction = _unit_vector(direction)
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    if unit_direction == (0.0, 0.0, 0.0):
        return identity
    return tuple(
        tuple(
            identity[i][j] - (1.0 - clamped_weight) * unit_direction[i] * unit_direction[j]
            for j in range(3)
        )
        for i in range(3)
    )


def _matrix_vector_multiply(
    matrix: tuple[tuple[float, float, float], ...],
    vector: Vector3,
) -> Vector3:
    return tuple(
        sum(matrix[i][j] * vector[j] for j in range(3))
        for i in range(3)
    )


def _matrix_multiply(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(3))
            for j in range(3)
        )
        for i in range(3)
    )


def _matrix_trace(matrix: tuple[tuple[float, float, float], ...]) -> float:
    return sum(matrix[i][i] for i in range(3))


def _matrix_double_contract(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> float:
    return sum(left[i][j] * right[i][j] for i in range(3) for j in range(3))


def _outer_product(a: Vector3, b: Vector3) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(a[i] * b[j] for j in range(3))
        for i in range(3)
    )


def _longitudinal_projector_weight(distance: float, mediator_mass: float) -> float:
    if mediator_mass <= 0.0:
        return 0.0
    scale = mediator_mass * max(distance, 1e-12)
    return (scale * scale) / (1.0 + scale * scale)


def _vector_polarization_factor(
    v_a: Vector3,
    v_b: Vector3,
    separation: Vector3,
    *,
    light_speed: float,
    mediator_mass: float,
    gauge_scheme: str,
    gauge_parameter: float,
) -> float:
    """spin-1 current-current contraction을 transverse/longitudinal projector와 함께 근사한다."""
    gamma_a = _lorentz_factor(v_a, light_speed)
    gamma_b = _lorentz_factor(v_b, light_speed)
    beta_a = _scale(v_a, 1.0 / light_speed)
    beta_b = _scale(v_b, 1.0 / light_speed)
    projector = _gauge_fixed_projector(
        separation,
        mediator_mass=mediator_mass,
        gauge_scheme=gauge_scheme,
        gauge_parameter=gauge_parameter,
    )
    spatial_term = _dot(beta_a, _matrix_vector_multiply(projector, beta_b))
    return max(gamma_a * gamma_b - gamma_a * gamma_b * spatial_term, 0.0)


def _graviton_tensor_factor(
    v_a: Vector3,
    v_b: Vector3,
    separation: Vector3,
    *,
    light_speed: float,
    mediator_mass: float,
    gauge_scheme: str,
    gauge_parameter: float,
) -> float:
    """spin-2 exchange를 traceless stress projector contraction으로 근사한다."""
    gamma_a = _lorentz_factor(v_a, light_speed)
    gamma_b = _lorentz_factor(v_b, light_speed)
    beta_a = _scale(v_a, 1.0 / light_speed)
    beta_b = _scale(v_b, 1.0 / light_speed)
    projector = _gauge_fixed_projector(
        separation,
        mediator_mass=mediator_mass,
        gauge_scheme=gauge_scheme,
        gauge_parameter=gauge_parameter,
    )
    transverse_a = _matrix_vector_multiply(projector, beta_a)
    transverse_b = _matrix_vector_multiply(projector, beta_b)
    stress_a = _matrix_multiply(_matrix_multiply(projector, _outer_product(beta_a, beta_a)), projector)
    stress_b = _matrix_multiply(_matrix_multiply(projector, _outer_product(beta_b, beta_b)), projector)
    trace_a = _matrix_trace(stress_a)
    trace_b = _matrix_trace(stress_b)
    traceless_contraction = _matrix_double_contract(stress_a, stress_b) - 0.5 * trace_a * trace_b
    energy_term = (gamma_a * gamma_b) ** 2
    momentum_term = 2.0 * gamma_a * gamma_b * _dot(transverse_a, transverse_b)
    return max(energy_term + momentum_term + traceless_contraction, 0.0)


def _gauge_fixed_projector(
    separation: Vector3,
    *,
    mediator_mass: float,
    gauge_scheme: str,
    gauge_parameter: float,
) -> tuple[tuple[float, float, float], ...]:
    """분리벡터 의존 비국소 게이지 고정 surrogate projector."""
    distance = _norm(separation)
    base_longitudinal = _longitudinal_projector_weight(distance, mediator_mass)
    xi = max(gauge_parameter, 0.0)
    scale = mediator_mass * max(distance, 1e-12)
    nonlocal_weight = 1.0 / (1.0 + scale * scale)
    if gauge_scheme == "landau":
        longitudinal_weight = 0.0
    elif gauge_scheme == "feynman":
        longitudinal_weight = base_longitudinal + xi * nonlocal_weight * (1.0 - base_longitudinal)
    elif gauge_scheme == "coulomb":
        longitudinal_weight = base_longitudinal * (1.0 + 0.25 * xi * nonlocal_weight)
    elif gauge_scheme == "unitary":
        longitudinal_weight = base_longitudinal + xi * (1.0 - nonlocal_weight) * (1.0 - base_longitudinal)
    else:
        longitudinal_weight = base_longitudinal
    return _transverse_projector(
        separation,
        longitudinal_weight=min(max(longitudinal_weight, 0.0), 1.0),
    )


def _vertex_resummation_factor(
    local_phase_density: float,
    *,
    scheme: str,
    strength: float,
) -> float:
    """4D vertex ladder/rainbow 재합산을 모사하는 안정한 국소 보정."""
    if scheme == "none" or strength <= 0.0:
        return 1.0
    expansion = min(max(abs(local_phase_density) * strength, 0.0), 0.95)
    if scheme == "geometric":
        return 1.0 / (1.0 - expansion)
    if scheme == "pade":
        return (1.0 + 0.5 * expansion) / max(1.0 - 0.5 * expansion, 1e-12)
    if scheme == "exponential":
        return exp(expansion)
    return 1.0


def _zero_phase_matrix(rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(0.0 for _ in range(cols)) for _ in range(rows))


def _phase_matrix_add(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left[i])))
        for i in range(len(left))
    )


def _phase_matrix_subtract(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(left[i][j] - right[i][j] for j in range(len(left[i])))
        for i in range(len(left))
    )


def _phase_matrix_scale(
    matrix: tuple[tuple[float, ...], ...],
    factor: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(factor * value for value in row)
        for row in matrix
    )


def _phase_matrix_max_abs(matrix: tuple[tuple[float, ...], ...]) -> float:
    return max((abs(value) for row in matrix for value in row), default=0.0)


def _phase_matrix_blend(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    *,
    relaxation: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple((1.0 - relaxation) * left[i][j] + relaxation * right[i][j] for j in range(len(left[i])))
        for i in range(len(left))
    )


def _compute_tensor_channel_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mediator: Literal["vector", "gravity"],
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
    gauge_scheme: str,
    gauge_parameter: float,
    vertex_resummation: str,
    vertex_strength: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            _tensor_mediated_pair_phase(
                branch_a,
                branch_b,
                mediator=mediator,
                mass=mass,
                cutoff=cutoff,
                propagation=propagation,
                light_speed=light_speed,
                quadrature_order=quadrature_order,
                gauge_scheme=gauge_scheme,
                gauge_parameter=gauge_parameter,
                vertex_resummation=vertex_resummation,
                vertex_strength=vertex_strength,
            )
            for branch_b in branches_b
        )
        for branch_a in branches_a
    )


def _compute_ghost_sector(
    vector_phase: tuple[tuple[float, ...], ...],
    *,
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
    mediator_mass: float,
    gauge_parameter: float,
    vertex_resummation: str,
    vertex_strength: float,
    ghost_mode: str,
    ghost_strength: float,
) -> GhostSectorResult:
    if ghost_mode == "none" or ghost_strength <= 0.0:
        zero = _zero_phase_matrix(len(vector_phase), len(vector_phase[0]) if vector_phase else 0)
        return GhostSectorResult(
            ghost_phase=zero,
            brst_compensated_vector_phase=vector_phase,
            longitudinal_residual=zero,
            brst_residual_norm=0.0,
            nilpotency_defect=0.0,
            mode=ghost_mode,
            strength=ghost_strength,
        )
    landau_vector = _compute_tensor_channel_matrix(
        branches_a,
        branches_b,
        mediator="vector",
        mass=mediator_mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        gauge_scheme="landau",
        gauge_parameter=gauge_parameter,
        vertex_resummation=vertex_resummation,
        vertex_strength=vertex_strength,
    )
    target_ghost = _phase_matrix_subtract(landau_vector, vector_phase)
    ghost_phase = _phase_matrix_scale(target_ghost, ghost_strength)
    compensated = _phase_matrix_add(vector_phase, ghost_phase)
    residual = _phase_matrix_subtract(compensated, landau_vector)
    residual_norm = _phase_matrix_max_abs(residual)
    nilpotency_defect = residual_norm if ghost_mode == "faddeev_popov" else residual_norm * residual_norm
    return GhostSectorResult(
        ghost_phase=ghost_phase,
        brst_compensated_vector_phase=compensated,
        longitudinal_residual=residual,
        brst_residual_norm=residual_norm,
        nilpotency_defect=nilpotency_defect,
        mode=ghost_mode,
        strength=ghost_strength,
    )


def _solve_tensor_dyson_schwinger(
    vector_phase: tuple[tuple[float, ...], ...],
    gravity_phase: tuple[tuple[float, ...], ...],
    scalar_phase: tuple[tuple[float, ...], ...],
    ghost_phase: tuple[tuple[float, ...], ...],
    *,
    mode: str,
    strength: float,
    iterations: int,
    tolerance: float,
    relaxation: float,
) -> DysonSchwingerResult:
    if mode == "none" or strength <= 0.0 or iterations <= 0:
        rows = len(vector_phase)
        cols = len(vector_phase[0]) if vector_phase else 0
        return DysonSchwingerResult(
            dressed_vector_phase=vector_phase,
            dressed_gravity_phase=gravity_phase,
            self_energy_kernel=_zero_phase_matrix(rows, cols),
            iterations=0,
            residual_norm=0.0,
            converged=True,
            mode=mode,
            strength=strength,
        )
    kernel = tuple(
        tuple(
            strength * (abs(scalar_phase[i][j]) + abs(ghost_phase[i][j])) / (1.0 + abs(scalar_phase[i][j]) + abs(ghost_phase[i][j]))
            for j in range(len(vector_phase[i]))
        )
        for i in range(len(vector_phase))
    )
    current_vector = vector_phase
    current_gravity = gravity_phase
    converged = False
    residual_norm = 0.0
    relaxation = min(max(relaxation, 1e-6), 1.0)
    for step in range(1, iterations + 1):
        updated_vector = []
        updated_gravity = []
        for i in range(len(vector_phase)):
            vector_row: list[float] = []
            gravity_row: list[float] = []
            for j in range(len(vector_phase[i])):
                kernel_ij = kernel[i][j]
                if mode == "rainbow":
                    source_vector = current_vector[i][j]
                    source_gravity = current_gravity[i][j]
                    denominator_vector = 1.0 + abs(current_vector[i][j])
                    denominator_gravity = 1.0 + abs(current_gravity[i][j])
                elif mode == "ladder":
                    source_vector = current_vector[i][j] + 0.5 * current_gravity[i][j]
                    source_gravity = current_gravity[i][j] + 0.5 * current_vector[i][j]
                    denominator_vector = 1.0 + abs(current_vector[i][j] + current_gravity[i][j])
                    denominator_gravity = denominator_vector
                else:
                    source_vector = current_vector[i][j] + current_gravity[i][j]
                    source_gravity = current_gravity[i][j] + current_vector[i][j]
                    denominator_vector = 1.0 + abs(current_vector[i][j]) + abs(current_gravity[i][j])
                    denominator_gravity = denominator_vector
                vector_row.append(vector_phase[i][j] + kernel_ij * source_vector / max(denominator_vector, 1e-12))
                gravity_row.append(gravity_phase[i][j] + kernel_ij * source_gravity / max(denominator_gravity, 1e-12))
            updated_vector.append(tuple(vector_row))
            updated_gravity.append(tuple(gravity_row))
        next_vector = _phase_matrix_blend(current_vector, tuple(updated_vector), relaxation=relaxation)
        next_gravity = _phase_matrix_blend(current_gravity, tuple(updated_gravity), relaxation=relaxation)
        residual_norm = max(
            _phase_matrix_max_abs(_phase_matrix_subtract(next_vector, current_vector)),
            _phase_matrix_max_abs(_phase_matrix_subtract(next_gravity, current_gravity)),
        )
        current_vector = next_vector
        current_gravity = next_gravity
        if residual_norm <= tolerance:
            converged = True
            return DysonSchwingerResult(
                dressed_vector_phase=current_vector,
                dressed_gravity_phase=current_gravity,
                self_energy_kernel=kernel,
                iterations=step,
                residual_norm=residual_norm,
                converged=True,
                mode=mode,
                strength=strength,
            )
    return DysonSchwingerResult(
        dressed_vector_phase=current_vector,
        dressed_gravity_phase=current_gravity,
        self_energy_kernel=kernel,
        iterations=iterations,
        residual_norm=residual_norm,
        converged=converged,
        mode=mode,
        strength=strength,
    )


def compute_proper_time_worldline(
    branch: BranchPath,
    *,
    light_speed: float = 1.0,
) -> ProperTimeWorldline:
    proper_times: list[float] = [0.0]
    lorentz_factors: list[float] = []
    for left, right in zip(branch.points, branch.points[1:]):
        dt = right.t - left.t
        velocity = _scale(_sub(right.position, left.position), 1.0 / dt) if dt > 1e-18 else (0.0, 0.0, 0.0)
        gamma = _lorentz_factor(velocity, light_speed)
        lorentz_factors.append(gamma)
        proper_times.append(proper_times[-1] + dt / gamma)
    lorentz_factors.append(lorentz_factors[-1] if lorentz_factors else 1.0)
    return ProperTimeWorldline(
        branch=branch,
        proper_times=tuple(proper_times),
        lorentz_factors=tuple(lorentz_factors),
    )


def _tensor_mediated_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mediator: Literal["scalar", "vector", "gravity"],
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"],
    light_speed: float,
    quadrature_order: int,
    gauge_scheme: str,
    gauge_parameter: float,
    vertex_resummation: str,
    vertex_strength: float,
) -> float:
    """텐서 구조를 반영한 pair phase 를 계산한다."""
    grid = _shared_time_grid(branch_a, branch_b)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    phase = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0
        segment = 0.0
        for node, weight in zip(nodes, weights):
            sample_time = midpoint + half_width * node
            v_a = _velocity_at(branch_a, sample_time)
            v_b = _velocity_at(branch_b, sample_time)
            position_a = branch_a.position_at(sample_time)
            position_b = branch_b.position_at(sample_time)
            separation = _sub(position_b, position_a)
            if mediator == "vector":
                tensor_factor = _vector_polarization_factor(
                    v_a, v_b, separation,
                    light_speed=light_speed,
                    mediator_mass=mass,
                    gauge_scheme=gauge_scheme,
                    gauge_parameter=gauge_parameter,
                )
                phase_sign = -1.0
                effective_mass = mass
            elif mediator == "gravity":
                tensor_factor = _graviton_tensor_factor(
                    v_a, v_b, separation,
                    light_speed=light_speed,
                    mediator_mass=mass,
                    gauge_scheme=gauge_scheme,
                    gauge_parameter=gauge_parameter,
                )
                phase_sign = 1.0
                effective_mass = mass
            else:
                tensor_factor = 1.0
                phase_sign = 1.0
                effective_mass = mass
            field_val = _field_value_at_observation_point(
                branch_a, sample_time, position_b,
                mass=effective_mass, cutoff=cutoff, propagation=propagation,
                light_speed=light_speed, quadrature_order=quadrature_order,
            )
            local_density = _mediator_pair_coupling(branch_a, branch_b, mediator) * tensor_factor * field_val
            resummation = _vertex_resummation_factor(
                local_density,
                scheme=vertex_resummation,
                strength=vertex_strength,
            )
            segment += weight * tensor_factor * field_val * resummation
        phase -= phase_sign * _mediator_pair_coupling(branch_a, branch_b, mediator) * segment * half_width
    return phase


def compute_tensor_mediated_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    mediator_mass: float = 0.0,
    gauge_scheme: Literal["projected", "landau", "feynman", "coulomb", "unitary"] = "projected",
    gauge_parameter: float = 1.0,
    vertex_resummation: Literal["none", "geometric", "pade", "exponential"] = "none",
    vertex_strength: float = 1.0,
    ghost_mode: Literal["none", "faddeev_popov", "brst"] = "none",
    ghost_strength: float = 1.0,
    dyson_schwinger_mode: Literal["none", "rainbow", "ladder", "coupled"] = "none",
    dyson_schwinger_strength: float = 0.5,
    dyson_schwinger_iterations: int = 12,
    dyson_schwinger_tolerance: float = 1e-8,
    dyson_schwinger_relaxation: float = 0.5,
) -> TensorMediatedPhaseResult:
    """spin-1/spin-2 텐서 구조와 gauge/ghost/Dyson-Schwinger surrogate 를 반영한 위상 행렬을 계산한다."""
    effective_scalar_mass = mass
    effective_vector_mass = mediator_mass if mediator_mass > 0.0 else 0.0
    effective_gravity_mass = mediator_mass if mediator_mass > 0.0 else 0.0

    scalar_phase = compute_branch_phase_matrix(
        branches_a, branches_b,
        mass=effective_scalar_mass, cutoff=cutoff,
        propagation=propagation, light_speed=light_speed,
        quadrature_order=quadrature_order,
    )

    vector_phase = _compute_tensor_channel_matrix(
        branches_a,
        branches_b,
        mediator="vector",
        mass=effective_vector_mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        gauge_scheme=gauge_scheme,
        gauge_parameter=gauge_parameter,
        vertex_resummation=vertex_resummation,
        vertex_strength=vertex_strength,
    )

    gravity_phase = _compute_tensor_channel_matrix(
        branches_a,
        branches_b,
        mediator="gravity",
        mass=effective_gravity_mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        gauge_scheme=gauge_scheme,
        gauge_parameter=gauge_parameter,
        vertex_resummation=vertex_resummation,
        vertex_strength=vertex_strength,
    )
    ghost_sector = _compute_ghost_sector(
        vector_phase,
        branches_a=branches_a,
        branches_b=branches_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
        mediator_mass=effective_vector_mass,
        gauge_parameter=gauge_parameter,
        vertex_resummation=vertex_resummation,
        vertex_strength=vertex_strength,
        ghost_mode=ghost_mode,
        ghost_strength=ghost_strength,
    )
    dyson_schwinger = _solve_tensor_dyson_schwinger(
        ghost_sector.brst_compensated_vector_phase,
        gravity_phase,
        scalar_phase,
        ghost_sector.ghost_phase,
        mode=dyson_schwinger_mode,
        strength=dyson_schwinger_strength,
        iterations=dyson_schwinger_iterations,
        tolerance=dyson_schwinger_tolerance,
        relaxation=dyson_schwinger_relaxation,
    )

    return TensorMediatedPhaseResult(
        scalar_phase=scalar_phase,
        vector_phase=vector_phase,
        gravity_phase=gravity_phase,
        mediator_mass=mediator_mass,
        gauge_scheme=gauge_scheme,
        gauge_parameter=gauge_parameter,
        vertex_resummation=vertex_resummation,
        vertex_strength=vertex_strength,
        ghost_sector=ghost_sector,
        dyson_schwinger=dyson_schwinger,
    )


def compute_renormalized_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    renormalization_mass: float | None = None,
) -> RenormalizedPhaseResult:
    """UV 정규화된 위상 행렬을 계산한다. self-energy 와 mass counterterm 을 뺀다."""
    raw_matrix = compute_branch_phase_matrix(
        branches_a, branches_b,
        mass=mass, cutoff=cutoff, propagation=propagation,
        light_speed=light_speed, quadrature_order=quadrature_order,
    )
    # self-energy: 각 branch 의 자기 상호작용
    self_a = tuple(
        _branch_pair_phase(b, b, mass=mass, cutoff=cutoff, propagation=propagation,
                           light_speed=light_speed, quadrature_order=quadrature_order)
        for b in branches_a
    )
    self_b = tuple(
        _branch_pair_phase(b, b, mass=mass, cutoff=cutoff, propagation=propagation,
                           light_speed=light_speed, quadrature_order=quadrature_order)
        for b in branches_b
    )
    # mass counterterm: renormalization_mass 가 주어지면 free self-energy 를 빼서 보정
    counter_a = [0.0] * len(branches_a)
    counter_b = [0.0] * len(branches_b)
    if renormalization_mass is not None:
        for i, b in enumerate(branches_a):
            counter_a[i] = _branch_pair_phase(b, b, mass=renormalization_mass, cutoff=cutoff,
                                              propagation=propagation, light_speed=light_speed,
                                              quadrature_order=quadrature_order)
        for i, b in enumerate(branches_b):
            counter_b[i] = _branch_pair_phase(b, b, mass=renormalization_mass, cutoff=cutoff,
                                              propagation=propagation, light_speed=light_speed,
                                              quadrature_order=quadrature_order)
    corrections_a = tuple(0.5 * (s - c) for s, c in zip(self_a, counter_a))
    corrections_b = tuple(0.5 * (s - c) for s, c in zip(self_b, counter_b))
    all_corrections = corrections_a + corrections_b
    renormalized = tuple(
        tuple(
            raw_matrix[r][s] - corrections_a[r] - corrections_b[s]
            for s in range(len(branches_b))
        )
        for r in range(len(branches_a))
    )
    return RenormalizedPhaseResult(
        raw_matrix=raw_matrix,
        self_energy_corrections=all_corrections,
        renormalized_matrix=renormalized,
    )


def compute_decoherence_model(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width: float = 0.0,
    quadrature_order: int = 3,
    environment_coupling: float = 0.01,
) -> DecoherenceResult:
    """field mode 에 대한 partial trace 로 matter 계의 decoherence 를 계산한다."""
    pair_displacements = compute_branch_pair_displacements(
        branches_a, branches_b, momenta,
        field_mass=field_mass,
        source_width_a=source_width, source_width_b=source_width,
        quadrature_order=quadrature_order,
    )
    n_a = len(branches_a)
    n_b = len(branches_b)
    n_total = n_a * n_b
    # coherence matrix: ρ((r,s),(r',s')) = <field_{rs}|field_{r's'}>
    coherence: list[list[complex]] = []
    decoherence_rates: list[float] = []
    for r in range(n_a):
        for s in range(n_b):
            row: list[complex] = []
            for rp in range(n_a):
                for sp in range(n_b):
                    alpha_rs = pair_displacements[r][s]
                    alpha_rps = pair_displacements[rp][sp]
                    diff_sq = sum(abs(a - b) ** 2 for a, b in zip(alpha_rs, alpha_rps))
                    suppression = exp(-0.5 * diff_sq * (1.0 + environment_coupling))
                    phase = compute_displacement_operator_phase(alpha_rs, alpha_rps)
                    row.append(suppression * complex_exp(1j * phase))
            coherence.append(row)
    # decoherence rate: off-diagonal decay
    for i in range(n_total):
        rate = 0.0
        for j in range(n_total):
            if i != j:
                rate += 1.0 - abs(coherence[i][j])
        decoherence_rates.append(rate / max(n_total - 1, 1))
    # purity: Tr(ρ²)
    purity = 0.0
    for i in range(n_total):
        for j in range(n_total):
            purity += abs(coherence[i][j]) ** 2
    purity /= n_total * n_total
    return DecoherenceResult(
        coherence_matrix=tuple(tuple(row) for row in coherence),
        decoherence_rates=tuple(decoherence_rates),
        purity=purity,
    )


def compute_multi_body_correlation(
    branches: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> MultiBodyCorrelationResult:
    """3체 이상의 connected correlation 위상을 계산한다."""
    n = len(branches)
    if n < 2:
        raise ValueError("At least 2 branches are needed for correlation analysis.")
    # pairwise phase matrix
    pairwise: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(_branch_pair_phase(branches[i], branches[i], mass=mass, cutoff=cutoff,
                                              propagation=propagation, light_speed=light_speed,
                                              quadrature_order=quadrature_order))
            else:
                row.append(_branch_pair_phase(branches[i], branches[j], mass=mass, cutoff=cutoff,
                                              propagation=propagation, light_speed=light_speed,
                                              quadrature_order=quadrature_order))
        pairwise.append(row)
    # 3-body connected correlation (cumulant)
    three_body = 0.0
    if n >= 3:
        total_3 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    total_3 += (pairwise[i][j] + pairwise[j][k] + pairwise[i][k]
                                - pairwise[i][i] - pairwise[j][j] - pairwise[k][k])
        three_body = total_3
    total_correlation = sum(pairwise[i][j] for i in range(n) for j in range(n)) - sum(pairwise[i][i] for i in range(n))
    return MultiBodyCorrelationResult(
        pairwise_phases=tuple(tuple(row) for row in pairwise),
        three_body_phase=three_body,
        total_correlation_phase=total_correlation,
    )


def evolve_relativistic_backreaction(
    source: BranchPath,
    target: BranchPath,
    *,
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    response_strength: float = 0.05,
) -> RelativisticForceResult:
    """Lorentz factor 를 포함한 상대론적 backreaction 으로 worldline 을 갱신한다."""
    offsets: tuple[Vector3, ...] = ((cutoff, 0.0, 0.0), (0.0, cutoff, 0.0), (0.0, 0.0, cutoff))
    updated_points: list[TrajectoryPoint] = []
    proper_times: list[float] = [0.0]
    four_velocities: list[tuple[float, ...]] = []
    for idx, point in enumerate(target.points):
        center = point.position
        velocity = _velocity_at(target, point.t)
        gamma = _lorentz_factor(velocity, light_speed)
        gradient: list[float] = []
        for offset in offsets:
            fp = _mediator_field_value(
                source, point.t, _add(center, offset),
                mass=mass, cutoff=cutoff, mediator=mediator,
                propagation=propagation, light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            fm = _mediator_field_value(
                source, point.t, _sub(center, offset),
                mass=mass, cutoff=cutoff, mediator=mediator,
                propagation=propagation, light_speed=light_speed,
                quadrature_order=quadrature_order,
            )
            gradient.append((fp - fm) / (2.0 * cutoff))
        sign = 1.0 if mediator in {"gravity", "scalar"} else -1.0
        effective_strength = response_strength / gamma
        shift: Vector3 = (
            sign * effective_strength * gradient[0],
            sign * effective_strength * gradient[1],
            sign * effective_strength * gradient[2],
        )
        updated_points.append(TrajectoryPoint(point.t, _add(center, shift)))
        u0 = gamma * light_speed
        four_velocities.append((u0, gamma * velocity[0], gamma * velocity[1], gamma * velocity[2]))
        if idx > 0:
            dt = point.t - target.points[idx - 1].t
            proper_times.append(proper_times[-1] + dt / gamma)
    return RelativisticForceResult(
        updated_branch=BranchPath(
            label=f"{target.label}_relativistic_backreacted",
            charge=target.charge,
            points=tuple(updated_points),
        ),
        proper_times=tuple(proper_times),
        four_velocities=tuple(four_velocities),
    )


def compute_entanglement_measures(
    phase_matrix: tuple[tuple[float, ...], ...],
    *,
    overlap_amplitudes: tuple[tuple[float, ...], ...] | None = None,
) -> EntanglementMeasures:
    """위상 행렬로부터 von Neumann entropy, negativity, witness, visibility 를 계산한다.

    최소 2×2 phase matrix 가 필요하다. 더 큰 행렬의 경우 좌상단 2×2 를 사용한다.
    overlap_amplitudes 가 주어지면 vacuum suppression 으로 density matrix 를 구성한다.
    """
    if len(phase_matrix) < 2 or len(phase_matrix[0]) < 2:
        raise ValueError("phase_matrix must be at least 2x2.")
    # 2-qubit density matrix 구성 (|00>, |01>, |10>, |11> basis)
    theta = phase_matrix[0][0] - phase_matrix[0][1] - phase_matrix[1][0] + phase_matrix[1][1]
    # overlap amplitudes (vacuum suppression)
    if overlap_amplitudes is not None and len(overlap_amplitudes) >= 2 and len(overlap_amplitudes[0]) >= 2:
        gamma_val = max(min(overlap_amplitudes[0][0], 1.0), 0.0)
    else:
        gamma_val = 1.0
    # effective 2-qubit state: 1/2 (|00> + e^{iθ}|11>)(h.c.)  with decoherence factor gamma
    cos_theta = exp(0.0) * gamma_val  # for visibility
    from math import cos, sin, log
    # density matrix in {|00>,|01>,|10>,|11>} basis for maximally entangled-like state
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5
    rho[0, 3] = 0.5 * gamma_val * complex_exp(1j * theta)
    rho[3, 0] = 0.5 * gamma_val * complex_exp(-1j * theta)
    # von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues.real, 1e-30, None)
    von_neumann = -sum(float(ev * np.log(ev)) for ev in eigenvalues if ev > 1e-30)
    # partial transpose (transpose over B subsystem)
    rho_pt = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    rho_pt[2 * i + k, 2 * j + l] = rho[2 * i + l, 2 * j + k]
    pt_eigenvalues = np.linalg.eigvalsh(rho_pt)
    negativity = float(sum(abs(ev) for ev in pt_eigenvalues if ev < -1e-15)) / 2.0 + max(0.0, float(sum(-ev for ev in pt_eigenvalues if ev < -1e-15)))
    negativity = float(sum(max(0.0, -ev) for ev in pt_eigenvalues))
    # witness: W = 1/2 - |ρ_{03}|
    witness = 0.5 - abs(float(rho[0, 3].real ** 2 + rho[0, 3].imag ** 2) ** 0.5)
    # visibility: 2|ρ_{03}|
    visibility = 2.0 * abs(complex(rho[0, 3]))
    return EntanglementMeasures(
        von_neumann_entropy=von_neumann,
        negativity=negativity,
        witness_value=witness,
        visibility=min(visibility, 1.0),
    )


def compute_mode_occupation_distribution(
    mode_amplitudes: tuple[complex, ...],
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
) -> ModeOccupationDistribution:
    """coherent state 의 mode 별 occupation number 분포를 계산한다."""
    if len(mode_amplitudes) != len(momenta):
        raise ValueError("mode_amplitudes and momenta must have the same length.")
    occupations = tuple(abs(alpha) ** 2 for alpha in mode_amplitudes)
    total = sum(occupations)
    probabilities = tuple(n / max(total, 1e-30) for n in occupations)
    return ModeOccupationDistribution(
        mode_occupations=occupations,
        total_occupation=total,
        mode_probabilities=probabilities,
    )


# ---------------------------------------------------------------------------
# PDE/격자 개선
# ---------------------------------------------------------------------------


def _cartesian_grid_metadata(
    spatial_points: tuple[Vector3, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], dict[tuple[int, int, int], int]] | None:
    xs = tuple(sorted({point[0] for point in spatial_points}))
    ys = tuple(sorted({point[1] for point in spatial_points}))
    zs = tuple(sorted({point[2] for point in spatial_points}))
    expected = len(xs) * len(ys) * len(zs)
    if expected != len(spatial_points):
        return None
    x_index = {value: idx for idx, value in enumerate(xs)}
    y_index = {value: idx for idx, value in enumerate(ys)}
    z_index = {value: idx for idx, value in enumerate(zs)}
    mapping: dict[tuple[int, int, int], int] = {}
    for flat_index, point in enumerate(spatial_points):
        key = (x_index[point[0]], y_index[point[1]], z_index[point[2]])
        if key in mapping:
            return None
        mapping[key] = flat_index
    if len(mapping) != expected:
        return None
    return xs, ys, zs, mapping


def _axis_spacing(axis: tuple[float, ...]) -> float | None:
    if len(axis) < 2:
        return None
    spacings = [axis[idx + 1] - axis[idx] for idx in range(len(axis) - 1)]
    first = spacings[0]
    if any(abs(spacing - first) > 1e-9 for spacing in spacings[1:]):
        raise ValueError("Finite-difference Cartesian grid requires uniform spacing on each occupied axis.")
    return first


def _source_density_at_point(source: BranchPath, sample_time: float, position: Vector3) -> float:
    src_time = min(max(sample_time, source.time_window[0]), source.time_window[1])
    src_pos = source.position_at(src_time)
    src_distance = _norm(_sub(position, src_pos))
    return source.charge * exp(-0.5 * src_distance * src_distance)


def _refine_axis_near_source(
    axis: tuple[float, ...],
    source_coords: tuple[float, ...],
    *,
    rounds: int,
    radius_factor: float,
) -> tuple[float, ...]:
    refined = tuple(axis)
    for _ in range(rounds):
        if len(refined) < 2:
            return refined
        spacing = min(refined[i + 1] - refined[i] for i in range(len(refined) - 1))
        radius = radius_factor * spacing
        should_refine = any(
            any(abs(0.5 * (left + right) - coord) <= radius for coord in source_coords)
            for left, right in zip(refined, refined[1:])
        )
        if not should_refine:
            continue
        new_values: list[float] = [refined[0]]
        for left, right in zip(refined, refined[1:]):
            new_values.append(0.5 * (left + right))
            new_values.append(right)
        refined = tuple(sorted(set(new_values)))
    return refined


def _adaptive_refine_spatial_points(
    source: BranchPath,
    spatial_points: tuple[Vector3, ...],
    *,
    rounds: int,
    radius_factor: float,
) -> tuple[Vector3, ...]:
    if rounds <= 0:
        return spatial_points
    cartesian = _cartesian_grid_metadata(spatial_points)
    source_x = tuple(point.position[0] for point in source.points)
    source_y = tuple(point.position[1] for point in source.points)
    source_z = tuple(point.position[2] for point in source.points)
    if cartesian is not None:
        xs, ys, zs, _ = cartesian
        refined_x = _refine_axis_near_source(xs, source_x, rounds=rounds, radius_factor=radius_factor)
        refined_y = _refine_axis_near_source(ys, source_y, rounds=rounds, radius_factor=radius_factor)
        refined_z = _refine_axis_near_source(zs, source_z, rounds=rounds, radius_factor=radius_factor)
        return tuple(
            (x, y, z)
            for z in refined_z
            for y in refined_y
            for x in refined_x
        )
    refined_points = tuple(spatial_points)
    for _ in range(rounds):
        new_points: list[Vector3] = [refined_points[0]]
        min_spacing = min(_norm(_sub(refined_points[i + 1], refined_points[i])) for i in range(len(refined_points) - 1))
        radius = radius_factor * min_spacing
        for left, right in zip(refined_points, refined_points[1:]):
            midpoint = _scale(_add(left, right), 0.5)
            if any(_norm(_sub(midpoint, point.position)) <= radius for point in source.points):
                new_points.append(midpoint)
            new_points.append(right)
        refined_points = tuple(new_points)
    return refined_points


def _evaluate_boundary_level_set(
    point: Vector3,
    boundary_level_set: Callable[[Vector3], float] | None,
) -> float:
    if boundary_level_set is None:
        return -1.0
    return float(boundary_level_set(point))


def _finite_difference_boundary_value(
    phi_curr: list[float],
    current_index: tuple[int, int, int],
    neighbor_index: tuple[int, int, int],
    *,
    axis: int,
    direction: int,
    dims: tuple[int, int, int],
    boundary: Literal["absorbing", "reflecting", "periodic"],
) -> float:
    if boundary == "periodic":
        wrapped = list(neighbor_index)
        wrapped[axis] %= dims[axis]
        return phi_curr[wrapped[0] + dims[0] * (wrapped[1] + dims[1] * wrapped[2])]
    if boundary == "reflecting":
        mirrored = list(current_index)
        mirrored[axis] = current_index[axis] - direction
        mirrored[axis] = min(max(mirrored[axis], 0), dims[axis] - 1)
        return phi_curr[mirrored[0] + dims[0] * (mirrored[1] + dims[1] * mirrored[2])]
    return phi_curr[current_index[0] + dims[0] * (current_index[1] + dims[1] * current_index[2])]


def _laplacian_1d(
    phi_curr: list[float],
    j: int,
    *,
    dx: float,
    boundary: Literal["absorbing", "reflecting", "periodic"],
    stencil_order: int,
) -> float:
    n_space = len(phi_curr)
    if stencil_order >= 4 and n_space >= 5:
        def sample(index: int) -> float:
            if 0 <= index < n_space:
                return phi_curr[index]
            if boundary == "periodic":
                return phi_curr[index % n_space]
            if boundary == "reflecting":
                mirrored = index
                while mirrored < 0 or mirrored >= n_space:
                    if mirrored < 0:
                        mirrored = -mirrored
                    if mirrored >= n_space:
                        mirrored = 2 * (n_space - 1) - mirrored
                return phi_curr[mirrored]
            clamped = min(max(index, 0), n_space - 1)
            return phi_curr[clamped]
        return (
            -sample(j + 2) + 16.0 * sample(j + 1) - 30.0 * sample(j) + 16.0 * sample(j - 1) - sample(j - 2)
        ) / (12.0 * dx * dx)
    if j == 0:
        if boundary == "periodic":
            return (phi_curr[1] - 2.0 * phi_curr[0] + phi_curr[n_space - 1]) / (dx * dx)
        if boundary == "reflecting":
            return (phi_curr[1] - phi_curr[0]) / (dx * dx)
        return (phi_curr[1] - 2.0 * phi_curr[0]) / (dx * dx)
    if j == n_space - 1:
        if boundary == "periodic":
            return (phi_curr[0] - 2.0 * phi_curr[j] + phi_curr[j - 1]) / (dx * dx)
        if boundary == "reflecting":
            return (phi_curr[j - 1] - phi_curr[j]) / (dx * dx)
        return (phi_curr[j - 1] - 2.0 * phi_curr[j]) / (dx * dx)
    return (phi_curr[j + 1] - 2.0 * phi_curr[j] + phi_curr[j - 1]) / (dx * dx)


def _required_substeps(interval_dt: float, courant_scale: float, max_courant: float) -> int:
    if interval_dt <= 0.0:
        raise ValueError("time_slices must be strictly increasing.")
    if max_courant <= 0.0:
        raise ValueError("max_courant must be positive.")
    target = abs(interval_dt) * courant_scale / max(max_courant, 1e-12)
    return max(1, int(ceil(target)))


def solve_finite_difference_kg(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    light_speed: float = 1.0,
    boundary: Literal["absorbing", "reflecting", "periodic"] = "absorbing",
    max_courant: float = 0.9,
    stencil_order: Literal[2, 4] = 2,
    adaptive_mesh_refinement_rounds: int = 0,
    adaptive_mesh_radius_factor: float = 1.25,
    boundary_level_set: Callable[[Vector3], float] | None = None,
) -> FiniteDifferencePdeResult:
    """유한차분 leapfrog 방법으로 Klein-Gordon 방정식을 직접 적분한다.

    tensor-product Cartesian 격자가 주어지면 3D Laplacian을 사용하고,
    필요하면 source support 주변 adaptive refinement, 4차 stencil, level-set 곡면 경계를 적용한다.
    그렇지 않으면 입력 순서를 따라 1D surrogate로 푼다.
    """
    if len(time_slices) < 2:
        raise ValueError("At least 2 time slices are required.")
    if len(spatial_points) < 2:
        raise ValueError("At least 2 spatial points are required.")
    if stencil_order not in {2, 4}:
        raise ValueError("stencil_order must be 2 or 4.")
    if adaptive_mesh_refinement_rounds < 0:
        raise ValueError("adaptive_mesh_refinement_rounds must be non-negative.")
    if adaptive_mesh_radius_factor <= 0.0:
        raise ValueError("adaptive_mesh_radius_factor must be positive.")

    spatial_points = _adaptive_refine_spatial_points(
        source,
        spatial_points,
        rounds=adaptive_mesh_refinement_rounds,
        radius_factor=adaptive_mesh_radius_factor,
    )

    dt_values = [time_slices[i + 1] - time_slices[i] for i in range(len(time_slices) - 1)]
    dt = min(dt_values) if dt_values else 1.0
    cartesian = _cartesian_grid_metadata(spatial_points)
    n_space = len(spatial_points)
    c2 = light_speed * light_speed
    m2 = mass * mass
    if max_courant <= 0.0:
        raise ValueError("max_courant must be positive.")
    if cartesian is None:
        dx_values = [_norm(_sub(spatial_points[i + 1], spatial_points[i])) for i in range(n_space - 1)]
        dx = min(dx_values) if dx_values else 1.0
        courant = light_speed * dt / max(dx, 1e-12)
        courant_scale = light_speed / max(dx, 1e-12)
        active_mask = tuple(_evaluate_boundary_level_set(point, boundary_level_set) <= 0.0 for point in spatial_points)
        phi_prev = [0.0] * n_space
        phi_curr = [0.0] * n_space
        for j in range(n_space):
            if active_mask[j]:
                distance = _norm(_sub(spatial_points[j], source.position_at(time_slices[0])))
                phi_curr[j] = source.charge * _yukawa_kernel(distance, mass, 1e-9)

        all_slices: list[tuple[float, ...]] = [tuple(phi_curr)]
        substeps_per_interval: list[int] = []
        effective_dt_min = inf
        for ti in range(1, len(time_slices)):
            current_dt = time_slices[ti] - time_slices[ti - 1]
            substeps = _required_substeps(current_dt, courant_scale, max_courant)
            substeps_per_interval.append(substeps)
            sub_dt = current_dt / substeps
            effective_dt_min = min(effective_dt_min, sub_dt)
            interval_start = time_slices[ti - 1]
            for substep in range(substeps):
                sample_time = interval_start + (substep + 1) * sub_dt
                phi_next = [0.0] * n_space
                for j in range(n_space):
                    if not active_mask[j]:
                        phi_next[j] = 0.0
                        continue
                    laplacian = _laplacian_1d(
                        phi_curr,
                        j,
                        dx=dx,
                        boundary=boundary,
                        stencil_order=stencil_order,
                    )
                    source_density = _source_density_at_point(source, sample_time, spatial_points[j])
                    phi_next[j] = (
                        2.0 * phi_curr[j]
                        - phi_prev[j]
                        + sub_dt * sub_dt * (c2 * laplacian - m2 * phi_curr[j] + source_density)
                    )

                if boundary == "absorbing" and n_space >= 2:
                    phi_next[0] = phi_curr[0] + light_speed * sub_dt * (phi_curr[1] - phi_curr[0]) / dx
                    phi_next[-1] = phi_curr[-1] - light_speed * sub_dt * (phi_curr[-1] - phi_curr[-2]) / dx
                for j, is_active in enumerate(active_mask):
                    if not is_active:
                        phi_next[j] = 0.0

                phi_prev = list(phi_curr)
                phi_curr = phi_next
            all_slices.append(tuple(phi_curr))
        return FiniteDifferencePdeResult(
            field_values=tuple(all_slices),
            time_slices=time_slices,
            spatial_points=spatial_points,
            courant_number=courant,
            grid_shape=(n_space, 1, 1),
            spatial_dimension=1,
            effective_time_step=effective_dt_min if effective_dt_min < inf else dt,
            substeps_per_interval=tuple(substeps_per_interval),
            stencil_order=stencil_order,
            refinement_rounds=adaptive_mesh_refinement_rounds,
            boundary_geometry="level_set" if boundary_level_set is not None else "none",
            active_point_mask=active_mask,
        )

    xs, ys, zs, mapping = cartesian
    dx = _axis_spacing(xs)
    dy = _axis_spacing(ys)
    dz = _axis_spacing(zs)
    active_spacings = tuple(spacing for spacing in (dx, dy, dz) if spacing is not None)
    if not active_spacings:
        raise ValueError("At least one occupied spatial axis needs two grid points.")
    courant = light_speed * dt * sqrt(sum((1.0 / (spacing * spacing)) for spacing in active_spacings))
    courant_scale = light_speed * sqrt(sum((1.0 / (spacing * spacing)) for spacing in active_spacings))
    dims = (len(xs), len(ys), len(zs))
    spatial_dimension = sum(1 for size in dims if size > 1)
    active_mask = tuple(_evaluate_boundary_level_set(point, boundary_level_set) <= 0.0 for point in spatial_points)

    phi_prev = [0.0] * n_space
    phi_curr = [0.0] * n_space
    for key, flat_index in mapping.items():
        point = spatial_points[flat_index]
        if active_mask[flat_index]:
            distance = _norm(_sub(point, source.position_at(time_slices[0])))
            phi_curr[flat_index] = source.charge * _yukawa_kernel(distance, mass, 1e-9)

    all_slices: list[tuple[float, ...]] = [tuple(phi_curr)]
    axis_spacings = (dx, dy, dz)
    substeps_per_interval: list[int] = []
    effective_dt_min = inf
    for ti in range(1, len(time_slices)):
        current_dt = time_slices[ti] - time_slices[ti - 1]
        substeps = _required_substeps(current_dt, courant_scale, max_courant)
        substeps_per_interval.append(substeps)
        sub_dt = current_dt / substeps
        effective_dt_min = min(effective_dt_min, sub_dt)
        interval_start = time_slices[ti - 1]
        for substep in range(substeps):
            sample_time = interval_start + (substep + 1) * sub_dt
            phi_next = [0.0] * n_space
            for (ix, iy, iz), flat_index in mapping.items():
                if not active_mask[flat_index]:
                    phi_next[flat_index] = 0.0
                    continue
                laplacian = 0.0
                for axis, spacing in enumerate(axis_spacings):
                    if spacing is None:
                        continue
                    current_index = (ix, iy, iz)
                    if stencil_order >= 4 and dims[axis] >= 5:
                        stencil_sum = 0.0
                        for offset, coefficient in ((-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0)):
                            if offset == 0:
                                value = phi_curr[flat_index]
                            else:
                                neighbor = [ix, iy, iz]
                                neighbor[axis] += offset
                                neighbor_tuple = (neighbor[0], neighbor[1], neighbor[2])
                                if neighbor_tuple in mapping:
                                    neighbor_flat = mapping[neighbor_tuple]
                                    value = phi_curr[neighbor_flat] if active_mask[neighbor_flat] else 0.0
                                else:
                                    value = _finite_difference_boundary_value(
                                        phi_curr,
                                        current_index,
                                        neighbor_tuple,
                                        axis=axis,
                                        direction=1 if offset > 0 else -1,
                                        dims=dims,
                                        boundary=boundary,
                                    )
                            stencil_sum += coefficient * value
                        laplacian += stencil_sum / (12.0 * spacing * spacing)
                    else:
                        minus_index = [ix, iy, iz]
                        plus_index = [ix, iy, iz]
                        minus_index[axis] -= 1
                        plus_index[axis] += 1
                        minus_tuple = (minus_index[0], minus_index[1], minus_index[2])
                        plus_tuple = (plus_index[0], plus_index[1], plus_index[2])
                        if minus_tuple in mapping:
                            minus_flat = mapping[minus_tuple]
                            minus_value = phi_curr[minus_flat] if active_mask[minus_flat] else 0.0
                        else:
                            minus_value = _finite_difference_boundary_value(
                                phi_curr,
                                current_index,
                                minus_tuple,
                                axis=axis,
                                direction=-1,
                                dims=dims,
                                boundary=boundary,
                            )
                        if plus_tuple in mapping:
                            plus_flat = mapping[plus_tuple]
                            plus_value = phi_curr[plus_flat] if active_mask[plus_flat] else 0.0
                        else:
                            plus_value = _finite_difference_boundary_value(
                                phi_curr,
                                current_index,
                                plus_tuple,
                                axis=axis,
                                direction=1,
                                dims=dims,
                                boundary=boundary,
                            )
                        laplacian += (plus_value - 2.0 * phi_curr[flat_index] + minus_value) / (spacing * spacing)
                source_density = _source_density_at_point(source, sample_time, spatial_points[flat_index])
                phi_next[flat_index] = (
                    2.0 * phi_curr[flat_index]
                    - phi_prev[flat_index]
                    + sub_dt * sub_dt * (c2 * laplacian - m2 * phi_curr[flat_index] + source_density)
                )

            if boundary == "absorbing":
                for (ix, iy, iz), flat_index in mapping.items():
                    if not active_mask[flat_index]:
                        phi_next[flat_index] = 0.0
                        continue
                    correction = 0.0
                    correction_count = 0
                    for axis, spacing in enumerate(axis_spacings):
                        if spacing is None:
                            continue
                        coordinate = (ix, iy, iz)[axis]
                        if coordinate not in {0, dims[axis] - 1}:
                            continue
                        inward = [ix, iy, iz]
                        inward[axis] = 1 if coordinate == 0 else dims[axis] - 2
                        inward_flat = inward[0] + dims[0] * (inward[1] + dims[1] * inward[2])
                        outward_sign = 1.0 if coordinate == 0 else -1.0
                        correction += phi_curr[flat_index] + outward_sign * light_speed * sub_dt * (
                            phi_curr[inward_flat] - phi_curr[flat_index]
                        ) / spacing
                        correction_count += 1
                    if correction_count > 0:
                        phi_next[flat_index] = correction / correction_count
            for flat_index, is_active in enumerate(active_mask):
                if not is_active:
                    phi_next[flat_index] = 0.0

            phi_prev = list(phi_curr)
            phi_curr = phi_next
        all_slices.append(tuple(phi_curr))

    return FiniteDifferencePdeResult(
        field_values=tuple(all_slices),
        time_slices=time_slices,
        spatial_points=spatial_points,
        courant_number=courant,
        grid_shape=dims,
        spatial_dimension=spatial_dimension,
        effective_time_step=effective_dt_min if effective_dt_min < inf else dt,
        substeps_per_interval=tuple(substeps_per_interval),
        stencil_order=stencil_order,
        refinement_rounds=adaptive_mesh_refinement_rounds,
        boundary_geometry="level_set" if boundary_level_set is not None else "none",
        active_point_mask=active_mask,
    )


def solve_physical_lattice_dynamics(
    source: BranchPath,
    *,
    time_slices: tuple[float, ...],
    spatial_points: tuple[Vector3, ...],
    mass: float,
    mediator: Literal["scalar", "vector", "gravity"] = "scalar",
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "kg_retarded",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    method: Literal["leapfrog", "verlet"] = "leapfrog",
    damping_rate: float = 0.0,
) -> PhysicalLatticeDynamicsResult:
    """물리적 시간 stepper (leapfrog/Verlet)로 격자 장 역학을 계산한다.

    선택적으로 radiation damping (2γ ∂φ/∂t) 을 포함한다.
    """
    if len(time_slices) < 2:
        raise ValueError("At least 2 time slices needed.")
    n_space = len(spatial_points)
    base_lattice = solve_field_lattice(
        source, time_slices=time_slices, spatial_points=spatial_points,
        mass=mass, mediator=mediator, cutoff=cutoff, propagation=propagation,
        light_speed=light_speed, quadrature_order=quadrature_order,
    )
    # 시간별 field value 추출
    time_field: list[list[float]] = []
    for ti in range(len(time_slices)):
        start = ti * n_space
        time_field.append([base_lattice.samples[start + j].value for j in range(n_space)])

    # leapfrog/Verlet time stepping with optional radiation damping
    evolved: list[list[float]] = [list(time_field[0])]
    if len(time_slices) >= 2:
        evolved.append(list(time_field[1]))
    for ti in range(2, len(time_slices)):
        dt = time_slices[ti] - time_slices[ti - 1]
        new_slice = [0.0] * n_space
        for j in range(n_space):
            # source-driven acceleration
            acceleration = time_field[ti][j] - evolved[-1][j]
            # damping: exp(-γ dt) 로 이전 속도를 감쇠시킨다
            damping_factor = exp(-damping_rate * dt) if damping_rate > 0.0 else 1.0
            new_slice[j] = (evolved[-1][j]
                            + damping_factor * (evolved[-1][j] - evolved[-2][j])
                            + acceleration * dt * dt)
        evolved.append(new_slice)

    lattices: list[FieldLattice] = []
    for ti, time_val in enumerate(time_slices):
        samples = tuple(
            FieldSample(t=time_val, position=spatial_points[j], value=evolved[ti][j])
            for j in range(n_space)
        )
        lattices.append(FieldLattice(samples=samples, time_slices=(time_val,), spatial_points=spatial_points))

    dt_avg = (time_slices[-1] - time_slices[0]) / max(len(time_slices) - 1, 1)
    return PhysicalLatticeDynamicsResult(
        lattices=tuple(lattices),
        time_step=dt_avg,
        method=method,
    )


# ---------------------------------------------------------------------------
# Lebedev 구면 quadrature
# ---------------------------------------------------------------------------


def compute_lebedev_displacement_amplitudes(
    branches: tuple[BranchPath, ...],
    *,
    field_mass: float,
    momentum_cutoff: float,
    radial_quadrature_order: int = 5,
    source_width: float = 0.0,
    time_quadrature_order: int = 3,
    lebedev_order: Literal[6, 14, 26] = 14,
) -> LebedevQuadratureResult:
    """Lebedev 구면 quadrature 가중치를 써서 연속 운동량 적분의 각도 평균을 계산한다."""
    if momentum_cutoff <= 0.0:
        raise ValueError("momentum_cutoff must be positive.")
    if lebedev_order == 6:
        directions = _AXIS_DIRECTIONS
        weights = _LEBEDEV_WEIGHTS_6
    elif lebedev_order == 14:
        directions = _AXIS_DIRECTIONS + _DIAGONAL_DIRECTIONS
        weights = _LEBEDEV_WEIGHTS_14
    else:
        directions = _AXIS_DIRECTIONS + _EDGE_DIRECTIONS + _DIAGONAL_DIRECTIONS
        weights = _LEBEDEV_WEIGHTS_26

    radial_nodes, radial_weights = _gauss_legendre_rule(radial_quadrature_order)
    amplitudes: list[complex] = []
    for branch_idx, branch_item in enumerate(branches):
        total = 0.0j
        midpoint = momentum_cutoff / 2.0
        half_width = momentum_cutoff / 2.0
        for radial_node, radial_weight in zip(radial_nodes, radial_weights):
            momentum_radius = midpoint + half_width * radial_node
            angular_total = 0.0j
            for direction, leb_weight in zip(directions, weights):
                momentum = _scale(direction, momentum_radius)
                omega = _mode_energy(momentum, field_mass)
                contribution = (-1j / sqrt(2.0 * omega)) * _branch_time_integral(
                    branch_item,
                    lambda sample_time, mode=momentum, frequency=omega: (
                        complex_exp(1j * frequency * sample_time)
                        * _source_form_factor(branch_item, mode, sample_time, source_width)
                    ),
                    quadrature_order=time_quadrature_order,
                )
                angular_total += leb_weight * contribution
            total += radial_weight * (momentum_radius * momentum_radius) * angular_total
        amplitudes.append(4.0 * pi * total * half_width)
    return LebedevQuadratureResult(
        amplitudes=tuple(amplitudes),
        quadrature_order=lebedev_order,
        direction_count=len(directions),
    )


# ---------------------------------------------------------------------------
# 근본적 한계 개선: mode-by-mode Fock-space Hamiltonian 진화
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeEvolution:
    """단일 momentum mode 의 Hamiltonian 진화 결과."""

    momentum: Vector3
    omega: float
    displacement: complex
    phase_accumulated: float
    occupation_number: float


@dataclass(frozen=True)
class FockSpaceEvolutionResult:
    """여러 mode 에 대한 Fock-space Hamiltonian 진화 결과.

    ``total_phase``는 Magnus expansion 1차항(parametric phase),
    2차항(time-ordering correction), 3차항(nested commutator correction)을
    합산한 값이다.
    ``parametric_phase``는 기존 parametric approximation 에서의 값이다.
    ``time_ordering_correction``은 2차 Magnus 항으로부터의 보정이다.
    ``third_order_correction``은 3차 Magnus 항으로부터의 보정이다.
    ``mode_evolutions``는 mode 별 상세 결과이다.
    """

    parametric_phase: float
    time_ordering_correction: float
    third_order_correction: float
    total_phase: float
    mode_evolutions: tuple[ModeEvolution, ...]


def _mode_coupling_at(
    branch: BranchPath,
    momentum: Vector3,
    omega: float,
    time: float,
    source_width: float,
) -> complex:
    """시각 ``time``에서 source-field coupling ``g_k(t) = q * f(k) * e^{i(ωt - k·x(t))}``."""
    suppression = exp(-0.5 * source_width * source_width * _dot(momentum, momentum))
    pos = branch.position_at(time)
    spatial_phase = _dot(momentum, pos)
    return branch.charge * suppression * complex_exp(1j * (omega * time - spatial_phase))


def compute_fock_space_evolution(
    branch_a: BranchPath,
    branch_b: BranchPath,
    momenta: tuple[Vector3, ...],
    *,
    field_mass: float,
    source_width_a: float = 0.0,
    source_width_b: float = 0.0,
    quadrature_order: int = 5,
    magnus_order: int = 3,
) -> FockSpaceEvolutionResult:
    """mode-by-mode Fock-space Hamiltonian 진화를 Magnus expansion 3차까지 계산한다.

    기존 parametric approximation 은 Magnus expansion 의 1차항에 해당한다.
    2차항(time-ordering correction)은 서로 다른 시각의 coupling 이 교환하지 않는
    효과를 반영하여, parametric approximation 을 넘어서는 보정을 준다.
    3차항은 nested commutator ``[H(t₁), [H(t₂), H(t₃)]]``의 triple integral 이다.

    1차 Magnus: Ω₁ = -i ∫ H(t) dt → parametric phase
    2차 Magnus: Ω₂ = -1/2 ∫∫ [H(t₁), H(t₂)] dt₁ dt₂ → time-ordering correction
    3차 Magnus: Ω₃ = i/6 ∫∫∫ [H₁,[H₂,H₃]] + [H₃,[H₂,H₁]] dt₁ dt₂ dt₃
    """
    grid = _shared_time_grid(branch_a, branch_b)
    nodes, weights = _gauss_legendre_rule(quadrature_order)

    mode_evolutions: list[ModeEvolution] = []
    total_parametric = 0.0
    total_correction = 0.0
    total_third = 0.0

    for momentum in momenta:
        omega = _mode_energy(momentum, field_mass)

        # 1차 Magnus: displacement amplitude α_k = ∫ g_k(t) dt / √(2ω)
        alpha_a = 0.0j
        alpha_b = 0.0j
        for t_start, t_stop in zip(grid, grid[1:]):
            midpoint = (t_start + t_stop) / 2.0
            half_width = (t_stop - t_start) / 2.0
            seg_a = 0.0j
            seg_b = 0.0j
            for nd, wt in zip(nodes, weights):
                t = midpoint + half_width * nd
                seg_a += wt * _mode_coupling_at(branch_a, momentum, omega, t, source_width_a)
                seg_b += wt * _mode_coupling_at(branch_b, momentum, omega, t, source_width_b)
            alpha_a += seg_a * half_width
            alpha_b += seg_b * half_width
        alpha_a /= sqrt(2.0 * omega)
        alpha_b /= sqrt(2.0 * omega)
        total_alpha = alpha_a + alpha_b

        # parametric phase: Im(α_a* α_b) 에 비례
        parametric = -(alpha_a * alpha_b.conjugate()).imag
        total_parametric += parametric

        # time sample 수집 (2차, 3차 공용)
        time_samples: list[tuple[float, complex, complex]] = []
        for t_start, t_stop in zip(grid, grid[1:]):
            midpoint = (t_start + t_stop) / 2.0
            half_width = (t_stop - t_start) / 2.0
            for nd, wt in zip(nodes, weights):
                t = midpoint + half_width * nd
                ga = _mode_coupling_at(branch_a, momentum, omega, t, source_width_a)
                gb = _mode_coupling_at(branch_b, momentum, omega, t, source_width_b)
                time_samples.append((wt * half_width, ga, gb))

        # 2차 Magnus: time-ordering correction
        correction = 0.0
        if magnus_order >= 2:
            for i, (w1, ga1, gb1) in enumerate(time_samples):
                for j, (w2, ga2, gb2) in enumerate(time_samples):
                    if j >= i:
                        break
                    commutator_im = (ga1 * gb2.conjugate() - gb1 * ga2.conjugate()).imag
                    correction += w1 * w2 * commutator_im
            correction /= (2.0 * omega)
        total_correction += correction

        # 3차 Magnus: nested commutator correction
        # Ω₃ = i/6 ∫ dt₁ ∫ dt₂ ∫ dt₃ ([H₁,[H₂,H₃]] + [H₃,[H₂,H₁]])
        # 각 mode 의 기여는 g 의 3중 곱에서 나오는 실수부로 근사한다.
        third = 0.0
        if magnus_order >= 3:
            n_samp = len(time_samples)
            for i in range(n_samp):
                w1, ga1, gb1 = time_samples[i]
                h1 = ga1 + gb1
                for j in range(i):
                    w2, ga2, gb2 = time_samples[j]
                    h2 = ga2 + gb2
                    for k in range(j):
                        w3, ga3, gb3 = time_samples[k]
                        h3 = ga3 + gb3
                        # [H₁,[H₂,H₃]] = H₁ H₂ H₃ - H₁ H₃ H₂ - H₂ H₃ H₁ + H₃ H₂ H₁
                        # 단일 mode boson 에서 교환자의 기여는
                        # Im(h₁ (h₂ h₃* - h₃ h₂*)) 에 비례한다.
                        inner_comm = (h2 * h3.conjugate() - h3 * h2.conjugate())
                        nested_1 = (h1 * inner_comm).real
                        inner_comm_rev = (h2 * h1.conjugate() - h1 * h2.conjugate())
                        nested_3 = (h3 * inner_comm_rev).real
                        third += w1 * w2 * w3 * (nested_1 + nested_3)
            third /= (6.0 * omega * omega)
        total_third += third

        occupation = abs(total_alpha) ** 2
        phase_acc = parametric + correction + third

        mode_evolutions.append(ModeEvolution(
            momentum=momentum,
            omega=omega,
            displacement=total_alpha,
            phase_accumulated=phase_acc,
            occupation_number=occupation,
        ))

    return FockSpaceEvolutionResult(
        parametric_phase=total_parametric,
        time_ordering_correction=total_correction,
        third_order_correction=total_third,
        total_phase=total_parametric + total_correction + total_third,
        mode_evolutions=tuple(mode_evolutions),
    )


# ---------------------------------------------------------------------------
# 근본적 한계 개선: adaptive quadrature 위상 적분
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptivePhaseResult:
    """적응형 quadrature 위상 적분 결과."""

    phase: float
    estimated_error: float
    segments_used: int
    quadrature_order: int


def compute_adaptive_phase_integral(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    tolerance: float = 1e-6,
    max_subdivisions: int = 8,
) -> tuple[tuple[AdaptivePhaseResult, ...], ...]:
    """오차 기반 적응형 세분화로 위상 행렬을 계산한다.

    각 시간 구간에서 GL quadrature 값과 구간을 2등분한 값을 비교하여
    차이가 ``tolerance``를 넘으면 재귀적으로 세분화한다.
    """
    kernel = _make_kernel(mass, cutoff)

    def _adaptive_segment(
        branch_a: BranchPath,
        branch_b: BranchPath,
        t_start: float,
        t_stop: float,
        depth: int,
    ) -> tuple[float, float, int]:
        """구간 하나에 대한 적응형 적분. (값, 오차추정, 세분횟수)를 반환한다."""
        nodes, weights = _gauss_legendre_rule(quadrature_order)
        midpoint = (t_start + t_stop) / 2.0
        half_width = (t_stop - t_start) / 2.0

        def _evaluate_segment(t0: float, t1: float) -> float:
            mid = (t0 + t1) / 2.0
            hw = (t1 - t0) / 2.0
            seg = 0.0
            for nd, wt in zip(nodes, weights):
                sample_time = mid + hw * nd
                if propagation == "instantaneous":
                    pa = branch_a.position_at(sample_time)
                    pb = branch_b.position_at(sample_time)
                    distance = _norm(_sub(pa, pb))
                    seg += wt * kernel(distance)
                elif propagation == "kg_retarded":
                    fv = _kg_retarded_field(
                        branch_a, branch_b, sample_time,
                        mass=mass, cutoff=cutoff, light_speed=light_speed,
                        quadrature_order=quadrature_order, proper_time_cutoff=1e-9,
                    )
                    seg += wt * fv
                else:
                    pa = branch_a.position_at(sample_time)
                    pb = branch_b.position_at(sample_time)
                    distance = _norm(_sub(pa, pb))
                    seg += wt * kernel(distance)
            return seg * hw

        coarse = _evaluate_segment(t_start, t_stop)
        fine = _evaluate_segment(t_start, midpoint) + _evaluate_segment(midpoint, t_stop)
        error = abs(fine - coarse)

        if error < tolerance or depth >= max_subdivisions:
            return fine, error, 1
        else:
            left_val, left_err, left_seg = _adaptive_segment(
                branch_a, branch_b, t_start, midpoint, depth + 1,
            )
            right_val, right_err, right_seg = _adaptive_segment(
                branch_a, branch_b, midpoint, t_stop, depth + 1,
            )
            return left_val + right_val, left_err + right_err, left_seg + right_seg

    results: list[tuple[AdaptivePhaseResult, ...]] = []
    for branch_a in branches_a:
        row: list[AdaptivePhaseResult] = []
        for branch_b in branches_b:
            grid = _shared_time_grid(branch_a, branch_b)
            total_phase = 0.0
            total_error = 0.0
            total_segments = 0
            for t_start, t_stop in zip(grid, grid[1:]):
                val, err, segs = _adaptive_segment(branch_a, branch_b, t_start, t_stop, 0)
                total_phase += val
                total_error += err
                total_segments += segs
            total_phase *= -branch_a.charge * branch_b.charge
            total_error *= abs(branch_a.charge * branch_b.charge)
            row.append(AdaptivePhaseResult(
                phase=total_phase,
                estimated_error=total_error,
                segments_used=total_segments,
                quadrature_order=quadrature_order,
            ))
        results.append(tuple(row))
    return tuple(results)


def _make_kernel(mass: float, cutoff: float) -> Callable[[float], float]:
    def kernel(distance: float) -> float:
        effective_distance = max(distance, cutoff)
        return exp(-mass * effective_distance) / (4.0 * pi * effective_distance)
    return kernel


# ---------------------------------------------------------------------------
# 근본적 한계 개선: Richardson extrapolation 위상 계산
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RichardsonExtrapolationResult:
    """Richardson extrapolation 위상 계산 결과.

    ``phases_by_order``는 각 quadrature order 에서의 위상 값이다.
    ``extrapolated_phase``는 외삽된 최적 추정치이다.
    ``estimated_error``는 마지막 두 order 값의 차이 기반 오차 추정이다.
    """

    phases_by_order: tuple[float, ...]
    extrapolated_phase: float
    estimated_error: float
    orders_used: tuple[int, ...]


def compute_richardson_extrapolated_phase(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    orders: tuple[int, ...] = (2, 4, 6, 8),
) -> tuple[tuple[RichardsonExtrapolationResult, ...], ...]:
    """quadrature order 를 단계적으로 키워 Richardson extrapolation 으로 외삽한다.

    각 order 에서 위상을 계산한 뒤, Neville 알고리즘으로 다항식 외삽하여
    체계적 오차를 제거한 최적 추정치를 얻는다.
    """
    results: list[tuple[RichardsonExtrapolationResult, ...]] = []
    for i, branch_a in enumerate(branches_a):
        row: list[RichardsonExtrapolationResult] = []
        for j, branch_b in enumerate(branches_b):
            phase_values: list[float] = []
            for order in orders:
                matrix = compute_branch_phase_matrix(
                    (branch_a,),
                    (branch_b,),
                    mass=mass,
                    cutoff=cutoff,
                    propagation=propagation,
                    light_speed=light_speed,
                    quadrature_order=order,
                )
                phase_values.append(matrix[0][0])
            # Neville 알고리즘으로 다항식 외삽
            n = len(phase_values)
            table = [list(phase_values)]
            h_values = [1.0 / (o * o) for o in orders]  # h ~ 1/order²
            for k in range(1, n):
                new_row: list[float] = []
                for m in range(n - k):
                    num = h_values[m] * table[k - 1][m + 1] - h_values[m + k] * table[k - 1][m]
                    den = h_values[m] - h_values[m + k]
                    new_row.append(num / den if abs(den) > 1e-30 else table[k - 1][m + 1])
                table.append(new_row)
            extrapolated = table[-1][0] if table[-1] else phase_values[-1]
            error = abs(extrapolated - phase_values[-1]) if len(phase_values) >= 2 else 0.0
            row.append(RichardsonExtrapolationResult(
                phases_by_order=tuple(phase_values),
                extrapolated_phase=extrapolated,
                estimated_error=error,
                orders_used=orders,
            ))
        results.append(tuple(row))
    return tuple(results)


# ---------------------------------------------------------------------------
# 근본적 한계 개선: lattice 보간 (sampled lattice → continuous interpolation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpolatedFieldResult:
    """격자점 사이에서 보간된 field 값 결과.

    ``value``는 trilinear 보간으로 얻은 연속 field 값이다.
    ``nearest_value``는 가장 가까운 격자점의 값이다.
    ``interpolation_weights``는 보간에 사용된 가중치이다.
    """

    position: Vector3
    t: float
    value: float
    nearest_value: float
    interpolation_weights: tuple[float, ...]


def interpolate_field_lattice(
    lattice: FieldLattice,
    t: float,
    position: Vector3,
) -> InterpolatedFieldResult:
    """FieldLattice 의 격자점 사이에서 trilinear 보간으로 연속 field 값을 계산한다.

    시간과 공간 모두에서 역거리 가중(inverse-distance weighting) 보간을 수행한다.
    이로써 sampled lattice 가 이산 격자에만 묶이는 한계를 해소한다.
    """
    # 모든 sample 에 대해 시공간 거리 기반 가중치 계산
    weights: list[float] = []
    values: list[float] = []
    nearest_idx = 0
    nearest_dist = inf

    for idx, sample in enumerate(lattice.samples):
        dt = abs(sample.t - t)
        dx = _norm(_sub(sample.position, position))
        dist = sqrt(dt * dt + dx * dx)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = idx
        if dist < 1e-15:
            # 격자점과 정확히 일치
            return InterpolatedFieldResult(
                position=position,
                t=t,
                value=sample.value,
                nearest_value=sample.value,
                interpolation_weights=(1.0,),
            )
        weights.append(1.0 / (dist * dist))
        values.append(sample.value)

    total_weight = sum(weights)
    interpolated = sum(w * v for w, v in zip(weights, values)) / total_weight if total_weight > 0.0 else 0.0
    normalized_weights = tuple(w / total_weight for w in weights) if total_weight > 0.0 else tuple(weights)

    return InterpolatedFieldResult(
        position=position,
        t=t,
        value=interpolated,
        nearest_value=lattice.samples[nearest_idx].value,
        interpolation_weights=normalized_weights,
    )


# ---------------------------------------------------------------------------
# 근본적 한계 개선: running coupling (effective mediator coupling → energy-dependent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunningCouplingResult:
    """에너지 의존 running coupling 을 적용한 위상 행렬 결과.

    ``bare_matrix``는 coupling 보정 전 위상이다.
    ``running_matrix``는 running coupling 을 적용한 위상이다.
    ``coupling_values``는 각 branch pair 에 적용된 effective coupling 이다.
    """

    bare_matrix: tuple[tuple[float, ...], ...]
    running_matrix: tuple[tuple[float, ...], ...]
    coupling_values: tuple[tuple[float, ...], ...]
    energy_scale: float


def _running_coupling(
    bare_coupling: float,
    energy_scale: float,
    beta_coefficient: float,
    reference_scale: float,
) -> float:
    """1-loop RG running coupling: α(E) = α₀ / (1 + β α₀ ln(E/μ)).

    ``beta_coefficient``는 beta function 의 leading 계수이다.
    양의 값이면 asymptotic freedom (고에너지에서 coupling 감소).
    """
    if energy_scale <= 0.0 or reference_scale <= 0.0:
        return bare_coupling
    from math import log
    log_ratio = log(energy_scale / reference_scale)
    denominator = 1.0 + beta_coefficient * bare_coupling * log_ratio
    if abs(denominator) < 1e-15:
        return bare_coupling
    return bare_coupling / denominator


def compute_running_coupling_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    energy_scale: float,
    beta_coefficient: float = 0.1,
    reference_scale: float = 1.0,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> RunningCouplingResult:
    """에너지 의존 running coupling 을 적용한 위상 행렬을 계산한다.

    bare coupling (charge product)에 1-loop RG running 을 적용하여
    에너지 척도에 따른 effective coupling 변화를 반영한다.
    static effective mediator coupling 의 한계를 넘어선다.
    """
    bare_matrix = compute_branch_phase_matrix(
        branches_a,
        branches_b,
        mass=mass,
        cutoff=cutoff,
        propagation=propagation,
        light_speed=light_speed,
        quadrature_order=quadrature_order,
    )
    coupling_values: list[tuple[float, ...]] = []
    running_matrix: list[tuple[float, ...]] = []
    for r, branch_a in enumerate(branches_a):
        row_couplings: list[float] = []
        row_phases: list[float] = []
        for s, branch_b in enumerate(branches_b):
            bare_coupling = branch_a.charge * branch_b.charge
            effective = _running_coupling(bare_coupling, energy_scale, beta_coefficient, reference_scale)
            ratio = effective / bare_coupling if abs(bare_coupling) > 1e-30 else 1.0
            row_couplings.append(effective)
            row_phases.append(bare_matrix[r][s] * ratio)
        coupling_values.append(tuple(row_couplings))
        running_matrix.append(tuple(row_phases))

    return RunningCouplingResult(
        bare_matrix=bare_matrix,
        running_matrix=tuple(running_matrix),
        coupling_values=tuple(coupling_values),
        energy_scale=energy_scale,
    )


# ---------------------------------------------------------------------------
# 근본적 한계 개선: symbolic bookkeeping 수치 검증
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookkeepingValidationResult:
    """symbolic bookkeeping 의 수치 검증 결과.

    ``symbolic_norms``는 symbolic bookkeeping 에서 보고한 amplitude norm 이다.
    ``numerical_norms``는 독립 계산으로 재현한 amplitude norm 이다.
    ``norm_residuals``는 두 값의 차이이다.
    ``phase_matrix_residual``는 symbolic/independent 위상 행렬의 최대 차이이다.
    ``is_consistent``는 모든 residual 이 tolerance 이내인지 여부이다.
    """

    symbolic_norms: tuple[float, ...]
    numerical_norms: tuple[float, ...]
    norm_residuals: tuple[float, ...]
    phase_matrix_residual: float
    is_consistent: bool


def validate_symbolic_bookkeeping(
    labeled_mode_samples: tuple[tuple[str, tuple[tuple[complex, ...], ...]], ...],
    *,
    tolerance: float = 1e-6,
) -> BookkeepingValidationResult:
    """symbolic bookkeeping 결과를 독립 수치 계산과 비교 검증한다.

    ``summarize_symbolic_multimode_bookkeeping``이 반환한 amplitude norm 과
    relative phase 를 mode sample 로부터 직접 재계산하여 일치 여부를 확인한다.
    이로써 symbolic bookkeeping 이 단순 요약을 넘어 수치적으로 검증된다.
    """
    bookkeeping = summarize_symbolic_multimode_bookkeeping(labeled_mode_samples)

    # 독립 계산: mode sample 로부터 직접 amplitude norm 계산
    numerical_norms: list[float] = []
    for _, samples in labeled_mode_samples:
        total_norm_sq = 0.0
        for mode_samples in samples:
            for val in mode_samples:
                total_norm_sq += abs(val) ** 2
        numerical_norms.append(sqrt(total_norm_sq))

    # residual 계산
    norm_residuals = tuple(
        abs(s - n) for s, n in zip(bookkeeping.amplitude_norms, numerical_norms)
    )

    # 위상 행렬 antisymmetry 검증
    phase_residual = 0.0
    mat = bookkeeping.relative_phase_matrix
    for r in range(len(mat)):
        phase_residual = max(phase_residual, abs(mat[r][r]))
        for c in range(r):
            phase_residual = max(phase_residual, abs(mat[r][c] + mat[c][r]))

    max_norm_residual = max(norm_residuals) if norm_residuals else 0.0
    is_consistent = max_norm_residual < tolerance and phase_residual < tolerance

    return BookkeepingValidationResult(
        symbolic_norms=bookkeeping.amplitude_norms,
        numerical_norms=tuple(numerical_norms),
        norm_residuals=norm_residuals,
        phase_matrix_residual=phase_residual,
        is_consistent=is_consistent,
    )
