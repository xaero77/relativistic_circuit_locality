from __future__ import annotations

"""스칼라장 모형에서 분기 궤적을 비교하는 유틸리티."""

from cmath import exp as complex_exp
from functools import lru_cache
from dataclasses import dataclass
from math import exp, inf, pi, sinh, sqrt
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


WidthSpec = float | tuple[float, ...]


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


def _pair_phase_integral(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    propagation: Literal["instantaneous", "retarded", "time_symmetric"],
    light_speed: float,
    quadrature_order: int,
    kernel: Callable[[float], float],
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
    propagation: Literal["instantaneous", "retarded", "time_symmetric"],
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


def compute_branch_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[tuple[float, ...], ...]:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if propagation not in {"instantaneous", "retarded", "time_symmetric"}:
        raise ValueError("propagation must be 'instantaneous', 'retarded', or 'time_symmetric'.")
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


def compute_displacement_operator_phase(
    left: tuple[complex, ...],
    right: tuple[complex, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("Displacement profiles must have the same number of momentum modes.")
    commutator = sum(alpha * beta.conjugate() - alpha.conjugate() * beta for alpha, beta in zip(left, right))
    return 0.5 * commutator.imag


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


def analyze_branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
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
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
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


def compute_wavepacket_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    widths_a: WidthSpec,
    widths_b: WidthSpec,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
    radial_quadrature_order: int = 5,
    radial_subdivisions: int = 24,
) -> tuple[tuple[float, ...], ...]:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if propagation not in {"instantaneous", "retarded", "time_symmetric"}:
        raise ValueError("propagation must be 'instantaneous', 'retarded', or 'time_symmetric'.")
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
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
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
    propagation: Literal["instantaneous", "retarded", "time_symmetric"] = "instantaneous",
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
