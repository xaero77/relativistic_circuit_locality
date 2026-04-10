from _shared import *


class EntanglementDiagnosticTests(unittest.TestCase):

    def test_entanglement_measures_returns_all_quantities(self) -> None:
        source = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        target = branch("B0", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        source2 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        target2 = branch("B1", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((source, source2), (target, target2), mass=0.5)
        result = compute_entanglement_measures(matrix)
        self.assertGreaterEqual(result.von_neumann_entropy, 0.0)
        self.assertGreaterEqual(result.negativity, 0.0)
        self.assertLessEqual(result.witness_value, 0.5)
        self.assertGreaterEqual(result.visibility, 0.0)
        self.assertLessEqual(result.visibility, 1.0)

    def test_entanglement_visibility_is_maximal_for_equal_separation(self) -> None:
        s0 = branch("A0", 1.0, [(0.0, (-2.0, 0.0, 0.0)), (2.0, (-2.0, 0.0, 0.0))])
        s1 = branch("A1", 1.0, [(0.0, (-1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))])
        t0 = branch("B0", 1.0, [(0.0, (1.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
        t1 = branch("B1", 1.0, [(0.0, (2.0, 0.0, 0.0)), (2.0, (2.0, 0.0, 0.0))])
        matrix = compute_branch_phase_matrix((s0, s1), (t0, t1), mass=0.5)
        result = compute_entanglement_measures(matrix)
        self.assertGreater(result.visibility, 0.0)

    def test_mode_occupation_distribution_sums_correctly(self) -> None:
        amplitudes = (0.5 + 0.3j, 0.1 - 0.2j, 0.4 + 0.0j)
        momenta = MOMENTA_AXIAL_3
        result = compute_mode_occupation_distribution(amplitudes, momenta, field_mass=0.5)
        self.assertEqual(len(result.mode_occupations), 3)
        self.assertAlmostEqual(result.total_occupation, sum(result.mode_occupations))
        self.assertAlmostEqual(sum(result.mode_probabilities), 1.0)

