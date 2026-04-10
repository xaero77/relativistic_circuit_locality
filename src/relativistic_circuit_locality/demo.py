from __future__ import annotations

"""Backward-compatible wrapper around the example scenarios package."""

from .examples import run_all


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
