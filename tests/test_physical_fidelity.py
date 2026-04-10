from _shared import *


class PhysicalFidelityTests(unittest.TestCase):

    def test_proper_time_worldline_static_branch(self) -> None:
        b = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        result = compute_proper_time_worldline(b, light_speed=1.0)
        self.assertAlmostEqual(result.proper_times[-1], 2.0)
        self.assertAlmostEqual(result.lorentz_factors[0], 1.0)

    def test_proper_time_worldline_moving_branch_has_time_dilation(self) -> None:
        b = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = compute_proper_time_worldline(b, light_speed=1.0)
        self.assertLess(result.proper_times[-1], 2.0)
        self.assertGreater(result.lorentz_factors[0], 1.0)

    def test_tensor_mediated_phase_returns_all_three_mediators(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous",
        )
        self.assertEqual(len(result.scalar_phase), 1)
        self.assertEqual(len(result.vector_phase), 1)
        self.assertEqual(len(result.gravity_phase), 1)
        self.assertNotEqual(result.scalar_phase[0][0], 0.0)

    def test_tensor_mediated_massive_mediator_differs(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        massless = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        massive = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=1.0,
        )
        self.assertNotAlmostEqual(massless.vector_phase[0][0], massive.vector_phase[0][0])

    def test_tensor_mediated_vector_projector_distinguishes_motion_direction(self) -> None:
        source_parallel = branch("A_parallel", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target_parallel = branch("B_parallel", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))])
        source_transverse = branch("A_transverse", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 1.0, 0.0))])
        target_transverse = branch("B_transverse", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 1.0, 0.0))])
        parallel = compute_tensor_mediated_phase_matrix(
            (source_parallel,), (target_parallel,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        transverse = compute_tensor_mediated_phase_matrix(
            (source_transverse,), (target_transverse,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        self.assertNotAlmostEqual(parallel.vector_phase[0][0], transverse.vector_phase[0][0])

    def test_tensor_mediated_gravity_projector_distinguishes_motion_direction(self) -> None:
        source_parallel = branch("A_parallel", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target_parallel = branch("B_parallel", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (3.0, 0.0, 0.0))])
        source_transverse = branch("A_transverse", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 1.0, 0.0))])
        target_transverse = branch("B_transverse", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 1.0, 0.0))])
        parallel = compute_tensor_mediated_phase_matrix(
            (source_parallel,), (target_parallel,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        transverse = compute_tensor_mediated_phase_matrix(
            (source_transverse,), (target_transverse,), mass=0.5, propagation="instantaneous", mediator_mass=0.0,
        )
        self.assertNotAlmostEqual(parallel.gravity_phase[0][0], transverse.gravity_phase[0][0])

    def test_tensor_mediated_gauge_scheme_changes_vector_phase(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.5, 0.5, 0.0))])
        landau = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="landau",
        )
        feynman = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="feynman",
        )
        self.assertNotAlmostEqual(landau.vector_phase[0][0], feynman.vector_phase[0][0])
        self.assertEqual(feynman.gauge_scheme, "feynman")

    def test_tensor_mediated_vertex_resummation_changes_tensor_phase(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.5, 0.0, 0.0)), (2.0, (-0.5, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.5, 0.0, 0.0)), (2.0, (0.5, 0.0, 0.0))])
        bare = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.2, propagation="instantaneous", vertex_resummation="none",
        )
        resum = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.2,
            propagation="instantaneous",
            vertex_resummation="pade",
            vertex_strength=0.75,
        )
        self.assertNotAlmostEqual(bare.vector_phase[0][0], resum.vector_phase[0][0])
        self.assertEqual(resum.vertex_resummation, "pade")

    def test_tensor_mediated_brst_ghost_sector_restores_landau_slice(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.5, 0.5, 0.0))])
        bare = compute_tensor_mediated_phase_matrix(
            (source,), (target,), mass=0.5, propagation="instantaneous", mediator_mass=0.8, gauge_scheme="feynman",
        )
        brst = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.5,
            propagation="instantaneous",
            mediator_mass=0.8,
            gauge_scheme="feynman",
            ghost_mode="brst",
            ghost_strength=1.0,
        )
        self.assertNotAlmostEqual(brst.ghost_sector.ghost_phase[0][0], 0.0)
        self.assertNotAlmostEqual(brst.ghost_sector.brst_compensated_vector_phase[0][0], bare.vector_phase[0][0])
        self.assertAlmostEqual(brst.ghost_sector.brst_residual_norm, 0.0, places=12)

    def test_tensor_mediated_dyson_schwinger_dresses_channels(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-1.5, 0.0, 0.0)), (2.0, (-0.5, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.5, 0.0, 0.0)), (2.0, (0.5, 0.0, 0.0))])
        result = compute_tensor_mediated_phase_matrix(
            (source,),
            (target,),
            mass=0.2,
            propagation="instantaneous",
            ghost_mode="brst",
            dyson_schwinger_mode="coupled",
            dyson_schwinger_strength=0.3,
            dyson_schwinger_iterations=16,
            dyson_schwinger_tolerance=1e-7,
            dyson_schwinger_relaxation=0.6,
        )
        self.assertGreater(result.dyson_schwinger.iterations, 0)
        self.assertTrue(result.dyson_schwinger.converged)
        self.assertNotAlmostEqual(
            result.dyson_schwinger.dressed_vector_phase[0][0],
            result.ghost_sector.brst_compensated_vector_phase[0][0],
        )
        self.assertNotAlmostEqual(
            result.dyson_schwinger.dressed_gravity_phase[0][0],
            result.gravity_phase[0][0],
        )

    def test_renormalized_phase_matrix_subtracts_self_energy(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1,
        )
        self.assertNotEqual(result.raw_matrix[0][0], result.renormalized_matrix[0][0])
        self.assertGreater(len(result.self_energy_corrections), 0)

    def test_renormalized_with_mass_counterterm(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        without = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1,
        )
        with_counter = compute_renormalized_phase_matrix(
            (source,), (target,), mass=0.5, cutoff=0.1, renormalization_mass=0.0,
        )
        self.assertNotAlmostEqual(without.self_energy_corrections[0], with_counter.self_energy_corrections[0])

    def test_decoherence_model_returns_coherence_and_purity(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        momenta = MOMENTA_AXIAL_2
        result = compute_decoherence_model(
            (source,), (target,), momenta, field_mass=0.5, environment_coupling=0.1,
        )
        self.assertGreater(result.purity, 0.0)
        self.assertLessEqual(result.purity, 1.0)
        self.assertEqual(len(result.decoherence_rates), 1)
        self.assertEqual(result.thermal_occupations, (0.0, 0.0))
        self.assertEqual(result.spectral_weights, (1.0, 1.0))
        self.assertEqual(result.memory_kernel_norm, 0.0)
        self.assertEqual(result.generated_lindblad_rates, tuple())
        self.assertEqual(result.detailed_balance_deviation, 0.0)
        self.assertEqual(len(result.renormalized_transition_energies), 1)
        self.assertEqual(result.lamb_shift_matrix, ((0.0 + 0.0j,),))
        self.assertEqual(result.bath_dressing_matrix, ((0.0 + 0.0j,),))
        self.assertEqual(result.bath_dressing_norm, 0.0)
        self.assertEqual(result.influence_phase_matrix, ((0.0,),))
        self.assertEqual(result.non_gaussian_cumulant_norm, 0.0)
        self.assertEqual(result.influence_iterations, 1)
        self.assertEqual(result.influence_residual, 0.0)
        self.assertTrue(result.influence_converged)
        self.assertAlmostEqual(result.lindblad_trace, 1.0)
        self.assertEqual(result.influence_kernel_mode, "surrogate")
        self.assertEqual(result.feynman_vernon_noise_matrix, ((0.0,),))
        self.assertEqual(result.feynman_vernon_dissipation_matrix, ((0.0,),))

    def test_decoherence_model_thermal_environment_suppresses_off_diagonal(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        momenta = MOMENTA_AXIAL_2
        vacuum = compute_decoherence_model(
            (source0, source1), (target,), momenta, field_mass=0.5,
        )
        thermal = compute_decoherence_model(
            (source0, source1), (target,), momenta, field_mass=0.5, environment_temperature=0.8,
        )
        self.assertGreater(thermal.thermal_occupations[0], 0.0)
        self.assertLess(abs(thermal.coherence_matrix[0][1]), abs(vacuum.coherence_matrix[0][1]))
        self.assertLess(thermal.purity, vacuum.purity)

    def test_decoherence_model_bath_spectral_density_reweights_modes(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.2, 0.0, 0.0), (1.5, 0.0, 0.0)),
            field_mass=0.5,
            environment_temperature=0.9,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega * omega),
        )
        self.assertGreater(result.spectral_weights[0], result.spectral_weights[1])
        self.assertGreater(result.thermal_occupations[0], result.thermal_occupations[1])

    def test_decoherence_model_lindblad_evolution_preserves_trace(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.7,),
            lindblad_time=1.5,
            lindblad_steps=96,
        )
        self.assertAlmostEqual(result.lindblad_trace, 1.0, places=8)
        self.assertGreater(result.coherence_matrix[0][0].real, result.coherence_matrix[1][1].real)
        self.assertLess(abs(result.coherence_matrix[0][1]), 0.5)

    def test_decoherence_model_non_markovian_memory_changes_evolution(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        markov = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.3,),
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        non_markov = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            lindblad_operators=(
                (
                    (0.0 + 0.0j, 1.0 + 0.0j),
                    (0.0 + 0.0j, 0.0 + 0.0j),
                ),
            ),
            lindblad_rates=(0.3,),
            lindblad_time=1.0,
            lindblad_steps=64,
            memory_kernel=lambda tau: exp(-2.0 * tau),
            memory_strength=0.4,
            colored_noise_correlation=lambda tau: exp(-tau),
            noise_time_window=1.0,
            noise_steps=32,
        )
        self.assertGreater(non_markov.memory_kernel_norm, 0.0)
        self.assertAlmostEqual(non_markov.lindblad_trace, 1.0, places=8)
        self.assertNotAlmostEqual(non_markov.purity, markov.purity)

    def test_decoherence_model_auto_generates_kms_lindblad_rates(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_temperature=0.7,
            bath_spectral_density=lambda omega: 1.0 + 0.5 * omega,
            auto_lindblad_from_bath=True,
            lindblad_time=1.0,
            lindblad_steps=64,
            dephasing_rate_scale=0.25,
            system_transition_energies=(0.0, 1.2),
        )
        self.assertGreater(len(result.generated_lindblad_rates), 0)
        self.assertLess(result.detailed_balance_deviation, 1e-10)
        self.assertAlmostEqual(result.lindblad_trace, 1.0, places=8)
        self.assertLess(result.purity, 1.0)

    def test_decoherence_model_auto_lindblad_covers_all_energy_level_pairs(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        source2 = branch("A2", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        temperature = 0.9
        result = compute_decoherence_model(
            (source0, source1, source2),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_coupling=0.2,
            environment_temperature=temperature,
            bath_spectral_density=lambda omega: 1.0 + 0.25 * omega,
            auto_lindblad_from_bath=True,
            system_transition_energies=(0.0, 0.7, 1.2),
            dephasing_rate_scale=0.1,
        )
        self.assertEqual(len(result.generated_lindblad_rates), 8)
        pair_gaps = (0.7, 1.2, 0.5)
        for pair_index, gap in enumerate(pair_gaps):
            emission_rate = result.generated_lindblad_rates[2 * pair_index]
            absorption_rate = result.generated_lindblad_rates[2 * pair_index + 1]
            self.assertGreater(emission_rate, absorption_rate)
            self.assertAlmostEqual(
                absorption_rate / emission_rate,
                exp(-gap / temperature),
                places=10,
            )
        self.assertLess(result.detailed_balance_deviation, 1e-10)

    def test_decoherence_model_auto_lamb_shift_renormalizes_energies(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        bare = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            system_transition_energies=(0.0, 1.2),
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        shifted = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            environment_temperature=0.7,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            system_transition_energies=(0.0, 1.2),
            auto_lamb_shift_from_bath=True,
            lamb_shift_strength=0.2,
            lamb_shift_cutoff=3.0,
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        self.assertNotAlmostEqual(
            shifted.renormalized_transition_energies[1],
            bare.renormalized_transition_energies[1],
        )
        self.assertNotEqual(shifted.lamb_shift_matrix[1][1], 0.0 + 0.0j)
        self.assertAlmostEqual(shifted.lindblad_trace, 1.0, places=8)

    def test_decoherence_model_auto_bath_dressing_builds_off_diagonal_self_energy(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        dressed = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            system_transition_energies=(0.0, 1.2),
            auto_bath_dressing_from_bath=True,
            bath_dressing_strength=0.3,
            lindblad_time=1.0,
            lindblad_steps=64,
        )
        self.assertGreater(dressed.bath_dressing_norm, 0.0)
        self.assertNotEqual(dressed.bath_dressing_matrix[0][1], 0.0 + 0.0j)
        self.assertAlmostEqual(dressed.lindblad_trace, 1.0, places=8)

    def test_decoherence_model_influence_functional_adds_non_gaussian_phase(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        base = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            colored_noise_correlation=lambda tau: exp(-tau),
            noise_time_window=1.0,
            noise_steps=32,
        )
        influenced = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            colored_noise_correlation=lambda tau: exp(-tau),
            influence_phase_strength=0.2,
            influence_time_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            non_gaussian_cumulant=lambda t1, t2: (t2 - t1) * (t1 + t2),
            non_gaussian_cumulant_strength=0.15,
        )
        self.assertGreater(abs(influenced.influence_phase_matrix[0][1]), 0.0)
        self.assertGreater(influenced.non_gaussian_cumulant_norm, 0.0)
        self.assertNotAlmostEqual(
            influenced.coherence_matrix[0][1].imag,
            base.coherence_matrix[0][1].imag,
        )

    def test_decoherence_model_self_consistent_influence_resummation_converges(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.5, 0.0, 0.0),),
            field_mass=0.5,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega),
            colored_noise_correlation=lambda tau: exp(-tau),
            influence_phase_strength=0.15,
            influence_iterations=16,
            influence_tolerance=1e-4,
            influence_relaxation=0.6,
        )
        self.assertGreaterEqual(result.influence_iterations, 1)
        self.assertLessEqual(result.influence_residual, 1e-4)
        self.assertTrue(result.influence_converged)

    def test_decoherence_model_feynman_vernon_kernel_adds_nonlocal_noise_and_phase(self) -> None:
        source0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        source1 = branch("A1", 1.0, [(0.0, (-0.8, 0.0, 0.0)), (2.0, (-0.2, 0.4, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (1.6, 0.3, 0.0))])
        surrogate = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.4, 0.0, 0.0), (0.8, 0.0, 0.0)),
            field_mass=0.5,
            environment_temperature=0.6,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega * omega),
            colored_noise_correlation=lambda tau: exp(-tau),
            influence_phase_strength=0.18,
            influence_time_grid=(0.0, 0.4, 0.8, 1.2),
        )
        feynman_vernon = compute_decoherence_model(
            (source0, source1),
            (target,),
            ((0.4, 0.0, 0.0), (0.8, 0.0, 0.0)),
            field_mass=0.5,
            environment_temperature=0.6,
            bath_spectral_density=lambda omega: 1.0 / (1.0 + omega * omega),
            influence_kernel_mode="feynman_vernon",
            influence_phase_strength=0.18,
            influence_time_grid=(0.0, 0.4, 0.8, 1.2),
            feynman_vernon_frequency_cutoff=2.5,
            feynman_vernon_frequency_samples=48,
            influence_iterations=4,
            influence_tolerance=1e-5,
        )
        self.assertEqual(feynman_vernon.influence_kernel_mode, "feynman_vernon")
        self.assertGreater(feynman_vernon.feynman_vernon_noise_matrix[0][1], 0.0)
        self.assertGreater(abs(feynman_vernon.influence_phase_matrix[0][1]), 0.0)
        self.assertNotAlmostEqual(
            abs(feynman_vernon.coherence_matrix[0][1]),
            abs(surrogate.coherence_matrix[0][1]),
        )
        self.assertNotAlmostEqual(
            feynman_vernon.coherence_matrix[0][1].imag,
            surrogate.coherence_matrix[0][1].imag,
        )

    def test_multi_body_correlation_pairwise_is_nonzero(self) -> None:
        a = branch("A", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        b = branch("B", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        c = branch("C", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        result = compute_multi_body_correlation((a, b, c), mass=0.5)
        self.assertNotEqual(result.total_correlation_phase, 0.0)
        self.assertNotEqual(result.three_body_phase, 0.0)
        self.assertEqual(len(result.pairwise_phases), 3)

    def test_relativistic_backreaction_returns_proper_times(self) -> None:
        source = branch("A0", 1.0, [(0.0, (0.0, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        result = evolve_relativistic_backreaction(
            source, target, mass=0.5, propagation="instantaneous", response_strength=0.1,
        )
        self.assertNotEqual(result.updated_branch.points[0].position, target.points[0].position)
        self.assertEqual(len(result.proper_times), 2)
        self.assertEqual(len(result.four_velocities), 2)

