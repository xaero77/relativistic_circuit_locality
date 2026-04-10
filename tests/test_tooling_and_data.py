"""Regression tests for tooling-facing helpers and lazy data loading."""

import unittest

from relativistic_circuit_locality.benchmarking import benchmark_representative_workloads, format_benchmark_report
from relativistic_circuit_locality.lebedev_tables import get_tabulated_lebedev_orders, get_tabulated_lebedev_table


class ToolingAndDataTests(unittest.TestCase):
    def test_tabulated_lebedev_orders_are_available(self) -> None:
        orders = get_tabulated_lebedev_orders()
        self.assertIn(38, orders)
        self.assertIn(194, orders)
        self.assertEqual(orders, tuple(sorted(orders)))

    def test_tabulated_lebedev_table_loads_from_package_data(self) -> None:
        table = get_tabulated_lebedev_table(38)
        self.assertIsNotNone(table)
        self.assertIn("0.009523809523810", table)

    def test_benchmark_suite_returns_named_results(self) -> None:
        results = benchmark_representative_workloads(iterations=1)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.average_seconds >= 0.0 for result in results))
        report = format_benchmark_report(results)
        self.assertIn("branch_phase_matrix", report)
        self.assertIn("finite_difference_kg", report)
