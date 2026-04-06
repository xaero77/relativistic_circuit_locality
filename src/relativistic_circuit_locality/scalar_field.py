from __future__ import annotations

"""Utilities for comparing branch trajectories through a scalar-field model."""

from dataclasses import dataclass
from math import exp, inf, pi, sqrt


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
        # Piecewise-linear interpolation only makes sense on an ordered timeline.
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
                # Interpolate within the active segment so mismatched sampling still works.
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
    # Merge all branch sample times so every trajectory segment boundary is respected.
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

    # Minimize the quadratic relative separation inside the segment bounds.
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
                    # A contact event inside the segment breaks field-only mediation.
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
    # Clamp very short distances to keep the kernel numerically well behaved.
    effective_distance = max(distance, cutoff)
    return exp(-mass * effective_distance) / (4.0 * pi * effective_distance)


def _branch_pair_phase(
    branch_a: BranchPath,
    branch_b: BranchPath,
    *,
    mass: float,
    cutoff: float,
) -> float:
    grid = _shared_time_grid(branch_a, branch_b)
    phase = 0.0
    for t_start, t_stop in zip(grid, grid[1:]):
        # Midpoint quadrature is enough because each segment is already piecewise linear.
        point_a = TrajectoryPoint(
            t=(t_start + t_stop) / 2.0,
            position=branch_a.position_at((t_start + t_stop) / 2.0),
        )
        point_b = TrajectoryPoint(
            t=(t_start + t_stop) / 2.0,
            position=branch_b.position_at((t_start + t_stop) / 2.0),
        )
        dt = t_stop - t_start
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
            # Report the nearest encounter across every branch pairing in the experiment.
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
