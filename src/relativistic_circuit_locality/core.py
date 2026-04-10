"""Stable entrypoints for the package's core locality model."""

from .geometry import BranchPath, SplineBranchPath, TrajectoryPoint, Vector3
from .scalar_field import (
    PairPhaseBreakdown,
    SimulationResult,
    compute_branch_phase_matrix,
    compute_closest_approach,
    compute_entanglement_phase,
    compute_spline_branch_phase_matrix,
    compute_wavepacket_phase_matrix,
    field_mediation_intervals,
    is_field_mediated,
    simulate,
)

__all__ = [
    "BranchPath",
    "PairPhaseBreakdown",
    "SimulationResult",
    "SplineBranchPath",
    "TrajectoryPoint",
    "Vector3",
    "compute_branch_phase_matrix",
    "compute_closest_approach",
    "compute_entanglement_phase",
    "compute_spline_branch_phase_matrix",
    "compute_wavepacket_phase_matrix",
    "field_mediation_intervals",
    "is_field_mediated",
    "simulate",
]
