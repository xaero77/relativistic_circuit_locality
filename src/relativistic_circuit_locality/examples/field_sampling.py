from __future__ import annotations

"""Examples around sampled fields, displacement amplitudes, and wavepackets."""

from ..experimental import (
    CompositeBranch,
    analyze_branch_pair_coherent_overlap,
    analyze_branch_pair_coherent_state,
    analyze_branch_pair_phase,
    compute_branch_displacement_amplitudes,
    compute_branch_pair_displacements,
    compute_composite_phase_matrix,
    compute_continuum_displacement_amplitudes,
    compute_displacement_operator_phase,
    compute_mediated_phase_matrix,
    compute_phi_rs_samples,
    compute_sampled_spacetime_phase,
    compute_wavepacket_phase_matrix,
    sample_branch_field,
)
from ._shared import base_branches


def collect_results() -> tuple[tuple[str, object], ...]:
    branches_a, branches_b = base_branches()
    sampled_field = sample_branch_field(
        branches_a[0],
        ((1.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0))),
        mass=0.5,
        propagation="kg_retarded",
    )
    phi_samples = compute_phi_rs_samples(
        branches_a[0],
        branches_b[0],
        sample_count=4,
        mass=0.5,
        propagation="kg_retarded",
    )
    sampled_phase = compute_sampled_spacetime_phase(
        branches_a[0],
        branches_b[0],
        mass=0.5,
        target_width=0.3,
        propagation="kg_retarded",
    )
    momenta = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0))
    displacement_a = compute_branch_displacement_amplitudes(
        branches_a,
        momenta,
        field_mass=0.5,
        source_width=0.2,
    )
    pair_displacements = compute_branch_pair_displacements(
        branches_a,
        branches_b,
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
    )
    coherent_state = analyze_branch_pair_coherent_state(
        branches_a[0],
        branches_b[0],
        momenta,
        field_mass=0.5,
        source_width_a=0.2,
        source_width_b=0.2,
        elapsed_time=1.5,
    )
    wavepacket_matrix = compute_wavepacket_phase_matrix(
        branches_a,
        branches_b,
        widths_a=(0.3, 0.5),
        widths_b=(0.3, 0.5),
        mass=0.5,
    )
    pair_breakdown = analyze_branch_pair_phase(
        branches_a[0],
        branches_b[0],
        mass=0.5,
        propagation="kg_retarded",
        cutoff=0.1,
    )
    return (
        ("sampled_field_A0", sampled_field),
        ("phi_rs_samples_A0_B0", phi_samples),
        ("sampled_spacetime_phase_A0_B0", sampled_phase),
        ("single_branch_displacements_A", displacement_a),
        ("pair_displacements", pair_displacements),
        (
            "continuum_displacements_A",
            compute_continuum_displacement_amplitudes(branches_a, field_mass=0.5, momentum_cutoff=1.0),
        ),
        (
            "pair_displacement_phase_A0B0_vs_A1B1",
            round(compute_displacement_operator_phase(pair_displacements[0][0], pair_displacements[1][1]), 6),
        ),
        (
            "pair_coherent_overlap_A0B0_vs_A1B1",
            analyze_branch_pair_coherent_overlap(
                (branches_a[0], branches_b[0]),
                (branches_a[1], branches_b[1]),
                momenta,
                field_mass=0.5,
                source_width_a=0.2,
                source_width_b=0.2,
            ),
        ),
        ("coherent_state_A0_B0", coherent_state),
        ("gravity_phase_matrix", compute_mediated_phase_matrix(branches_a, branches_b, mass=0.5, mediator="gravity")),
        (
            "composite_phase_matrix",
            compute_composite_phase_matrix(
                (CompositeBranch("A", branches_a),),
                (CompositeBranch("B", branches_b),),
                mass=0.5,
            ),
        ),
        ("wavepacket_phase_matrix", wavepacket_matrix),
        ("pair_breakdown_A0_B0", pair_breakdown),
    )
