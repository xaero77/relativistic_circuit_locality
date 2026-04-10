"""Developer-time shim for the src-layout package.

This file allows `python -m unittest` and ad-hoc local imports to resolve the
real package from `src/relativistic_circuit_locality` without requiring
`PYTHONPATH=src`.
"""

from __future__ import annotations

from pathlib import Path


_SRC_INIT = Path(__file__).resolve().parent.parent / "src" / "relativistic_circuit_locality" / "__init__.py"
__path__ = [str(_SRC_INIT.parent)]
__file__ = str(_SRC_INIT)

exec(compile(_SRC_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals())
