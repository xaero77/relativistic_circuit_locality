from __future__ import annotations

"""CLI entrypoint for example scenarios."""

import sys

from . import SCENARIOS, run_all, run_scenario


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args == ["all"]:
        run_all()
        return 0
    scenario = args[0]
    if scenario not in SCENARIOS:
        print("available scenarios:", ", ".join(("all",) + tuple(SCENARIOS)))
        return 1
    run_scenario(scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
