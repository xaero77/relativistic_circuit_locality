"""상대론적 회로 국소성 도우미의 공개 API를 내보낸다."""

from .scalar_field import (
    BranchPath,
    SimulationResult,
    TrajectoryPoint,
    compute_branch_phase_matrix,
    compute_closest_approach,
    compute_entanglement_phase,
    field_mediation_intervals,
    is_field_mediated,
    simulate,
)

__all__ = [
    "BranchPath",
    "SimulationResult",
    "TrajectoryPoint",
    "compute_branch_phase_matrix",
    "compute_closest_approach",
    "compute_entanglement_phase",
    "field_mediation_intervals",
    "is_field_mediated",
    "simulate",
]
