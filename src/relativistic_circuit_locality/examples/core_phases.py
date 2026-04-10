from __future__ import annotations

"""Core locality examples intended for first-time users."""

from ..core import compute_entanglement_phase, simulate
from ._shared import base_branches


def collect_results() -> tuple[tuple[str, object], ...]:
    branches_a, branches_b = base_branches()
    instantaneous = simulate(branches_a, branches_b, mass=0.5)
    retarded = simulate(branches_a, branches_b, mass=0.5, propagation="retarded")
    time_symmetric = simulate(branches_a, branches_b, mass=0.5, propagation="time_symmetric")
    causal_history = simulate(branches_a, branches_b, mass=0.5, propagation="causal_history")
    kg_retarded = simulate(branches_a, branches_b, mass=0.5, propagation="kg_retarded")
    return (
        ("closest_approach", round(instantaneous.closest_approach, 6)),
        ("mediation_intervals", instantaneous.mediation_intervals),
        ("instantaneous_phase_matrix", instantaneous.phase_matrix),
        ("retarded_phase_matrix", retarded.phase_matrix),
        ("time_symmetric_phase_matrix", time_symmetric.phase_matrix),
        ("causal_history_phase_matrix", causal_history.phase_matrix),
        ("kg_retarded_phase_matrix", kg_retarded.phase_matrix),
        (
            "instantaneous_relative_entangling_phase",
            round(compute_entanglement_phase(instantaneous.phase_matrix, 0, 1, 0, 1), 6),
        ),
        ("retarded_relative_entangling_phase", round(compute_entanglement_phase(retarded.phase_matrix, 0, 1, 0, 1), 6)),
        (
            "time_symmetric_relative_entangling_phase",
            round(compute_entanglement_phase(time_symmetric.phase_matrix, 0, 1, 0, 1), 6),
        ),
        (
            "causal_history_relative_entangling_phase",
            round(compute_entanglement_phase(causal_history.phase_matrix, 0, 1, 0, 1), 6),
        ),
        (
            "kg_retarded_relative_entangling_phase",
            round(compute_entanglement_phase(kg_retarded.phase_matrix, 0, 1, 0, 1), 6),
        ),
    )
