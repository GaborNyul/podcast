# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Post-implementation gate: all quality checks plus lockfile integrity."""

import argparse
from collections.abc import Sequence

import gatelib


def main(
    argv: Sequence[str] | None = None,
    runner: gatelib.CommandRunner = gatelib.run_command,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket_id", help="ticket or phase identifier being finished")
    args = parser.parse_args(argv)
    gates = list(gatelib.default_gates())
    lock_gate = gatelib.lock_check_gate()
    if lock_gate is not None:
        gates.append(lock_gate)
    results = gatelib.run_gates(gates, runner)
    return gatelib.report(results, f"post-ticket {args.ticket_id}")


if __name__ == "__main__":
    raise SystemExit(main())
