"""Regression tests for experiment specs and batch execution."""

import json
import tempfile
import unittest
from pathlib import Path

from relativistic_circuit_locality.experiments import (
    build_core_phase_preset,
    experiment_spec_from_dict,
    experiment_spec_to_dict,
    load_batch_specs,
    run_batch,
    run_experiment,
    save_batch_report,
)


class ExperimentPipelineTests(unittest.TestCase):
    def test_preset_runs_all_default_propagations(self) -> None:
        report = run_experiment(build_core_phase_preset())
        self.assertEqual(len(report.reports), 5)
        self.assertEqual(report.reports[0].propagation, "instantaneous")
        self.assertIsNotNone(report.reports[0].entangling_phase)

    def test_experiment_spec_roundtrip_preserves_name(self) -> None:
        spec = build_core_phase_preset()
        restored = experiment_spec_from_dict(experiment_spec_to_dict(spec))
        self.assertEqual(restored.name, spec.name)
        self.assertEqual(restored.mass, spec.mass)
        self.assertEqual(restored.propagations, spec.propagations)

    def test_batch_report_writes_json_and_csv(self) -> None:
        report = run_batch((build_core_phase_preset(),))
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = save_batch_report(report, tmpdir)
            self.assertTrue(summary_path.exists())
            self.assertTrue((Path(tmpdir) / "core_phase_demo.json").exists())
            self.assertIn("entangling_phase", summary_path.read_text(encoding="utf-8"))

    def test_load_batch_specs_accepts_batch_file(self) -> None:
        payload = {"experiments": [experiment_spec_to_dict(build_core_phase_preset(name="batch_case"))]}
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "batch.json"
            spec_path.write_text(json.dumps(payload), encoding="utf-8")
            specs = load_batch_specs(spec_path)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].name, "batch_case")
