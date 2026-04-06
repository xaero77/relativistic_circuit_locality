from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, pi, sqrt


Vector3 = tuple[float, float, float]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


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
        if len(self.points) < 2:
            raise ValueError("Each branch path needs at least two trajectory points.")
        times = [point.t for point in self.points]
        if any(t1 >= t2 for t1, t2 in zip(times, times[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")

    @property
    def time_window(self) -> tuple[float, float]:
        return (self.points[0].t, self.points[-1].t)


@dataclass(frozen=True)
class SimulationResult:
    closest_approach: float
    mediation_intervals: tuple[tuple[float, float], ...]
    phase_matrix: tuple[tuple[float, ...], ...]


def _validate_pair(a: BranchPath, b: BranchPath) -> None:
    if len(a.points) != len(b.points):
        raise ValueError("Compared branches must share the same temporal discretization.")
    for left, right in zip(a.points, b.points):
        if abs(left.t - right.t) > 1e-12:
            raise ValueError("Compared branches must share the same sample times.")


def _midpoint(a: TrajectoryPoint, b: TrajectoryPoint) -> TrajectoryPoint:
    return TrajectoryPoint(
        t=(a.t + b.t) / 2.0,
        position=(
            (a.position[0] + b.position[0]) / 2.0,
            (a.position[1] + b.position[1]) / 2.0,
            (a.position[2] + b.position[2]) / 2.0,
        ),
    )


def _pairwise_spacelike_margin(a: TrajectoryPoint, b: TrajectoryPoint, light_speed: float) -> float:
    spatial_distance = _norm(_sub(a.position, b.position))
    timelike_reach = light_speed * abs(a.t - b.t)
    return spatial_distance - timelike_reach


def compute_closest_approach(branch_a: BranchPath, branch_b: BranchPath) -> float:
    _validate_pair(branch_a, branch_b)
    return min(
        _norm(_sub(point_a.position, point_b.position))
        for point_a, point_b in zip(branch_a.points, branch_b.points)
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

    base_times = tuple(point.t for point in branches_a[0].points)
    for collection in (branches_a, branches_b):
        for branch in collection:
            if tuple(point.t for point in branch.points) != base_times:
                raise ValueError("All branches must share the same temporal discretization.")

    intervals: list[tuple[float, float]] = []
    current_start: float | None = None

    for index in range(len(base_times) - 1):
        segment_is_spacelike = True
        for branch_a in branches_a:
            for branch_b in branches_b:
                candidates = (
                    branch_a.points[index],
                    branch_a.points[index + 1],
                    _midpoint(branch_a.points[index], branch_a.points[index + 1]),
                )
                others = (
                    branch_b.points[index],
                    branch_b.points[index + 1],
                    _midpoint(branch_b.points[index], branch_b.points[index + 1]),
                )
                if any(
                    _pairwise_spacelike_margin(point_a, point_b, light_speed) <= tolerance
                    for point_a in candidates
                    for point_b in others
                ):
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
    effective_distance = max(distance, cutoff)
    return exp(-mass * effective_distance) / (4.0 * pi * effective_distance)


def _branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float,
) -> float:
    _validate_pair(branch_a, branch_b)
    phase = 0.0
    for index in range(len(branch_a.points) - 1):
        point_a = _midpoint(branch_a.points[index], branch_a.points[index + 1])
        point_b = _midpoint(branch_b.points[index], branch_b.points[index + 1])
        dt = branch_a.points[index + 1].t - branch_a.points[index].t
        distance = _norm(_sub(point_a.position, point_b.position))
        phase -= branch_a.charge * branch_b.charge * _yukawa_kernel(distance, mass, cutoff) * dt
    return phase


def compute_branch_phase_matrix(
    branches_a: tuple[BranchPath, ...],
    branches_b: tuple[BranchPath, ...],
    *,
    mass: float,
    cutoff: float = 1e-9,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            _branch_pair_phase(branch_a, branch_b, mass=mass, cutoff=cutoff)
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
) -> SimulationResult:
    closest_approach = inf
    for branch_a in branches_a:
        for branch_b in branches_b:
            closest_approach = min(closest_approach, compute_closest_approach(branch_a, branch_b))

    return SimulationResult(
        closest_approach=closest_approach,
        mediation_intervals=field_mediation_intervals(
            branches_a,
            branches_b,
            light_speed=light_speed,
            tolerance=tolerance,
        ),
        phase_matrix=compute_branch_phase_matrix(branches_a, branches_b, mass=mass, cutoff=cutoff),
    )
