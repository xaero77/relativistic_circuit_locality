from __future__ import annotations

"""Trajectory and interpolation primitives shared across the package."""

from dataclasses import dataclass
from math import sqrt


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
                weight = (t - left.t) / (right.t - left.t)
                return _add(left.position, _scale(_sub(right.position, left.position), weight))
        raise ValueError("Requested time could not be interpolated from trajectory points.")


def _cubic_spline_coefficients(
    times: tuple[float, ...],
    values: tuple[float, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    """Return natural cubic spline coefficients `(a, b, c, d)` for each segment."""

    n = len(times) - 1
    h = [times[i + 1] - times[i] for i in range(n)]
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i]) - (3.0 / h[i - 1]) * (values[i] - values[i - 1])
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
    return tuple((values[i], b_coeff[i], c_coeff[i], d_coeff[i]) for i in range(n))


@dataclass(frozen=True)
class SplineBranchPath:
    """Cubic-spline branch path compatible with the linear branch interface."""

    label: str
    charge: float
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Each branch path needs at least two trajectory points.")
        times = [point.t for point in self.points]
        if any(t1 >= t2 for t1, t2 in zip(times, times[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")
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
                return (
                    ax + bx * dt + ccx * dt * dt + dx * dt * dt * dt,
                    ay + by * dt + ccy * dt * dt + dy * dt * dt * dt,
                    az + bz * dt + ccz * dt * dt + dz * dt * dt * dt,
                )
        raise ValueError("Requested time could not be interpolated from trajectory points.")

    def as_branch_path(self) -> BranchPath:
        return BranchPath(label=self.label, charge=self.charge, points=self.points)

    def refined_branch_path(self, subdivisions: int = 4) -> BranchPath:
        new_points: list[TrajectoryPoint] = [self.points[0]]
        for left, right in zip(self.points, self.points[1:]):
            for k in range(1, subdivisions + 1):
                frac = k / subdivisions
                t = left.t + frac * (right.t - left.t)
                new_points.append(TrajectoryPoint(t, self.position_at(t)))
        return BranchPath(label=self.label, charge=self.charge, points=tuple(new_points))


__all__ = [
    "Vector3",
    "TrajectoryPoint",
    "BranchPath",
    "SplineBranchPath",
    "_add",
    "_dot",
    "_norm",
    "_scale",
    "_sub",
]
