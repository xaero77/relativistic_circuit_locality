# relativistic_circuit_locality

Python reference implementation of the core constructive pieces from `q-2026-03-24-2046.pdf`
("Circuit locality from relativistic locality in scalar field mediated entanglement", arXiv:2305.05645v4).

This repository implements the parts of the paper that translate cleanly into a small, dependency-free
numerical model:

- branch-wise trajectories for two quantum-controlled sources
- the spacelike-separation criterion behind the paper's field-mediation circuit
- a discrete closest-approach check corresponding to the paper's `d_min > 0` condition
- a quasi-static Yukawa approximation to the on-shell scalar-field phase `theta_rs = -1/2 ∫ rho_rs phi_rs`
- extraction of relative entangling phases between qudit branches

The code intentionally does not attempt a full QFT simulator. It stays within the paper's parametric
approximation and uses time-discretized trajectories plus a quasi-static interaction kernel as a practical
numerical surrogate.

## Layout

- `src/relativistic_circuit_locality/scalar_field.py`: main model and numerical helpers
- `src/relativistic_circuit_locality/demo.py`: minimal runnable example
- `tests/test_scalar_field.py`: unit tests
- `CHAT.md`: implementation notes, completed features, and remaining work

## Quick Start

Run the demo:

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
```

Run the tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
