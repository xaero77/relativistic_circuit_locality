from __future__ import annotations

"""스칼라장 모형에서 분기 궤적을 비교하는 유틸리티."""

from functools import lru_cache
from dataclasses import dataclass
from math import exp, inf, pi, sqrt
from typing import Literal


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


def _branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float,
    propagation: Literal["instantaneous", "retarded"],
    light_speed: float,
    quadrature_order: int,
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
            point_b = TrajectoryPoint(t=sample_time, position=branch_b.position_at(sample_time))
            if propagation == "instantaneous":
                point_a = TrajectoryPoint(t=sample_time, position=branch_a.position_at(sample_time))
            else:
                retarded_time = _retarded_source_time(
                    branch_a,
                    branch_b,
                    sample_time,
                    light_speed=light_speed,
                )
                if retarded_time is None:
                    continue
                point_a = TrajectoryPoint(t=retarded_time, position=branch_a.position_at(retarded_time))
            distance = _norm(_sub(point_a.position, point_b.position))
            segment_integral += weight * _yukawa_kernel(distance, mass, cutoff)
        phase -= branch_a.charge * branch_b.charge * segment_integral * half_width
    return phase


def compute_branch_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
    propagation: Literal["instantaneous", "retarded"] = "instantaneous",
    light_speed: float = 1.0,
    quadrature_order: int = 3,
) -> tuple[tuple[float, ...], ...]:
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive.")
    if propagation not in {"instantaneous", "retarded"}:
        raise ValueError("propagation must be either 'instantaneous' or 'retarded'.")
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
    propagation: Literal["instantaneous", "retarded"] = "instantaneous",
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
