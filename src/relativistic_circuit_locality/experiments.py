"""Reproducible experiment specs, batch execution, and result export."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

from .core import BranchPath, SimulationResult, TrajectoryPoint, Vector3, compute_entanglement_phase, simulate
from .examples._shared import base_branches

PropagationMode = Literal["instantaneous", "retarded", "time_symmetric", "causal_history", "kg_retarded"]
_DEFAULT_PROPAGATIONS: tuple[PropagationMode, ...] = (
    "instantaneous",
    "retarded",
    "time_symmetric",
    "causal_history",
    "kg_retarded",
)


@dataclass(frozen=True)
class TrajectoryPointSpec:
    t: float
    position: Vector3

    def to_point(self) -> TrajectoryPoint:
        return TrajectoryPoint(self.t, self.position)


@dataclass(frozen=True)
class BranchSpec:
    label: str
    charge: float
    points: tuple[TrajectoryPointSpec, ...]

    def to_branch_path(self) -> BranchPath:
        return BranchPath(
            label=self.label,
            charge=self.charge,
            points=tuple(point.to_point() for point in self.points),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    branches_a: tuple[BranchSpec, ...]
    branches_b: tuple[BranchSpec, ...]
    mass: float
    cutoff: float = 1e-9
    light_speed: float = 1.0
    tolerance: float = 1e-9
    quadrature_order: int = 3
    propagations: tuple[PropagationMode, ...] = _DEFAULT_PROPAGATIONS
    entanglement_indices: tuple[int, int, int, int] | None = (0, 1, 0, 1)


@dataclass(frozen=True)
class PropagationReport:
    propagation: PropagationMode
    simulation: SimulationResult
    entangling_phase: float | None
    max_abs_phase: float


@dataclass(frozen=True)
class ExperimentReport:
    spec: ExperimentSpec
    reports: tuple[PropagationReport, ...]


@dataclass(frozen=True)
class BatchReport:
    experiments: tuple[ExperimentReport, ...]


def _normalize_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize_json_value(item) for item in value]
    return value


def _branch_spec_from_dict(payload: dict[str, Any]) -> BranchSpec:
    points = tuple(
        TrajectoryPointSpec(
            t=float(point["t"]),
            position=tuple(float(component) for component in point["position"]),  # type: ignore[arg-type]
        )
        for point in payload["points"]
    )
    return BranchSpec(label=str(payload["label"]), charge=float(payload["charge"]), points=points)


def experiment_spec_from_dict(payload: dict[str, Any]) -> ExperimentSpec:
    propagations = tuple(payload.get("propagations", _DEFAULT_PROPAGATIONS))
    entanglement_indices = payload.get("entanglement_indices", (0, 1, 0, 1))
    return ExperimentSpec(
        name=str(payload["name"]),
        branches_a=tuple(_branch_spec_from_dict(branch) for branch in payload["branches_a"]),
        branches_b=tuple(_branch_spec_from_dict(branch) for branch in payload["branches_b"]),
        mass=float(payload["mass"]),
        cutoff=float(payload.get("cutoff", 1e-9)),
        light_speed=float(payload.get("light_speed", 1.0)),
        tolerance=float(payload.get("tolerance", 1e-9)),
        quadrature_order=int(payload.get("quadrature_order", 3)),
        propagations=tuple(str(mode) for mode in propagations),  # type: ignore[arg-type]
        entanglement_indices=None if entanglement_indices is None else tuple(int(index) for index in entanglement_indices),
    )


def experiment_spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    return _normalize_json_value(spec)


def load_batch_specs(path: str | Path) -> tuple[ExperimentSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "experiments" in payload:
        return tuple(experiment_spec_from_dict(item) for item in payload["experiments"])
    return (experiment_spec_from_dict(payload),)


def _phase_matrix_max_abs(matrix: tuple[tuple[float, ...], ...]) -> float:
    return max((abs(value) for row in matrix for value in row), default=0.0)


def run_experiment(spec: ExperimentSpec) -> ExperimentReport:
    branches_a = tuple(branch.to_branch_path() for branch in spec.branches_a)
    branches_b = tuple(branch.to_branch_path() for branch in spec.branches_b)
    reports: list[PropagationReport] = []
    for propagation in spec.propagations:
        simulation = simulate(
            branches_a,
            branches_b,
            mass=spec.mass,
            cutoff=spec.cutoff,
            light_speed=spec.light_speed,
            tolerance=spec.tolerance,
            propagation=propagation,
            quadrature_order=spec.quadrature_order,
        )
        entangling_phase = None
        if spec.entanglement_indices is not None:
            entangling_phase = compute_entanglement_phase(simulation.phase_matrix, *spec.entanglement_indices)
        reports.append(
            PropagationReport(
                propagation=propagation,
                simulation=simulation,
                entangling_phase=entangling_phase,
                max_abs_phase=_phase_matrix_max_abs(simulation.phase_matrix),
            )
        )
    return ExperimentReport(spec=spec, reports=tuple(reports))


def run_batch(specs: tuple[ExperimentSpec, ...]) -> BatchReport:
    return BatchReport(experiments=tuple(run_experiment(spec) for spec in specs))


def save_batch_report(report: BatchReport, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for experiment in report.experiments:
        experiment_path = output_path / f"{experiment.spec.name}.json"
        experiment_path.write_text(json.dumps(_normalize_json_value(experiment), indent=2) + "\n", encoding="utf-8")
        for propagation_report in experiment.reports:
            summary_rows.append(
                {
                    "experiment": experiment.spec.name,
                    "propagation": propagation_report.propagation,
                    "mass": experiment.spec.mass,
                    "closest_approach": propagation_report.simulation.closest_approach,
                    "interval_count": len(propagation_report.simulation.mediation_intervals),
                    "entangling_phase": propagation_report.entangling_phase,
                    "max_abs_phase": propagation_report.max_abs_phase,
                }
            )
    summary_path = output_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("experiment", "propagation", "mass", "closest_approach", "interval_count", "entangling_phase", "max_abs_phase"),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def _branch_spec_from_branch(branch: BranchPath) -> BranchSpec:
    return BranchSpec(
        label=branch.label,
        charge=branch.charge,
        points=tuple(TrajectoryPointSpec(t=point.t, position=point.position) for point in branch.points),
    )


def build_core_phase_preset(name: str = "core_phase_demo") -> ExperimentSpec:
    branches_a, branches_b = base_branches()
    return ExperimentSpec(
        name=name,
        branches_a=tuple(_branch_spec_from_branch(branch) for branch in branches_a),
        branches_b=tuple(_branch_spec_from_branch(branch) for branch in branches_b),
        mass=0.5,
        propagations=_DEFAULT_PROPAGATIONS,
        entanglement_indices=(0, 1, 0, 1),
    )


_PRESET_BUILDERS = {
    "core": build_core_phase_preset,
}


def _write_specs_file(specs: tuple[ExperimentSpec, ...], path: Path) -> None:
    payload: dict[str, Any]
    if len(specs) == 1:
        payload = experiment_spec_to_dict(specs[0])
    else:
        payload = {"experiments": [experiment_spec_to_dict(spec) for spec in specs]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a JSON spec or batch file.")
    run_parser.add_argument("spec_path")
    run_parser.add_argument("output_dir")

    preset_parser = subparsers.add_parser("preset", help="Materialize and run a built-in preset.")
    preset_parser.add_argument("preset_name", choices=tuple(sorted(_PRESET_BUILDERS)))
    preset_parser.add_argument("output_dir")

    args = parser.parse_args(argv)
    if args.command == "run":
        specs = load_batch_specs(args.spec_path)
        save_batch_report(run_batch(specs), args.output_dir)
        return 0
    preset_builder = _PRESET_BUILDERS[args.preset_name]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (preset_builder(),)
    _write_specs_file(specs, output_dir / "spec.json")
    save_batch_report(run_batch(specs), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
