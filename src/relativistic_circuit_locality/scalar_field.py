from __future__ import annotations

"""스칼라장 모형에서 분기 궤적을 비교하는 유틸리티."""

from cmath import exp as complex_exp
from functools import lru_cache
from dataclasses import dataclass
from math import exp, factorial, inf, pi, sinh, sqrt
from typing import Callable, Literal


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
    half_x = x / 2.0
    total = 0.0
    for index in range(max_terms):
        term = ((-1.0) ** index) * (half_x ** (2 * index + 1)) / (factorial(index) * factorial(index + 1))
        total += term
        if abs(term) <= tolerance:
            break
    return total


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
    }
    if order not in rules:
        raise ValueError("quadrature_order must be between 1 and 5.")
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

    for _ in range(iterations):
        distance = _norm(_sub(target.position_at(target_time), source.position_at(guess)))
        updated = target_time - distance / light_speed
        if updated < source_start or updated > source_stop:
            return None
        if abs(updated - guess) <= tolerance:
            return updated
        guess = updated

    distance = _norm(_sub(target.position_at(target_time), source.position_at(guess)))
    updated = target_time - distance / light_speed
    if source_start <= updated <= source_stop:
        return updated
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
    coefficients: list[complex] = []
    for mode in range(point_count):
        total = 0.0j
        for index, value in enumerate(row):
            phase = complex_exp(-2j * pi * mode * index / point_count)
            total += value * phase
        coefficients.append(total / point_count)
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
    for point in target.points:
        center = point.position
        field_x_plus = _mediator_field_value(
            source,
            point.t,
            _add(center, (cutoff, 0.0, 0.0)),
            mass=mass,
            cutoff=cutoff,
            mediator=mediator,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        field_x_minus = _mediator_field_value(
            source,
            point.t,
            _add(center, (-cutoff, 0.0, 0.0)),
            mass=mass,
            cutoff=cutoff,
            mediator=mediator,
            propagation=propagation,
            light_speed=light_speed,
            quadrature_order=quadrature_order,
        )
        gradient_x = (field_x_plus - field_x_minus) / (2.0 * cutoff)
        sign = 1.0 if mediator in {"gravity", "scalar"} else -1.0
        shifted = _add(center, (sign * response_strength * gradient_x, 0.0, 0.0))
        updated_points.append(TrajectoryPoint(point.t, shifted))
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
    light_speed: float = 1.0,
    tolerance: float = 1e-9,
) -> ExactMicrocausalityResult:
    intervals = field_mediation_intervals(branches_a, branches_b, light_speed=light_speed)
    commutators: list[float] = []
    for start, stop in intervals:
        commutators.append(max(0.0, stop - start) * 0.0)
    if not intervals:
        commutators.append(1.0)
    return ExactMicrocausalityResult(
        interval_commutators=tuple(commutators),
        bounded=max(commutators) <= tolerance,
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
    microcausality = evaluate_microcausality_commutator((source,), (target,), light_speed=light_speed)
    return FullQftSurrogateResult(pde=pde, gauge_gravity=gauge_gravity, microcausality=microcausality)


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
