"""Pre-implementation gate: verify a clean baseline before starting a ticket."""

import argparse
from collections.abc import Sequence

import gatelib


def main(
    argv: Sequence[str] | None = None,
    runner: gatelib.CommandRunner = gatelib.run_command,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket_id", help="ticket or phase identifier being started")
    args = parser.parse_args(argv)
    results = gatelib.run_gates(gatelib.default_gates(), runner)
    return gatelib.report(results, f"pre-ticket {args.ticket_id}")


if __name__ == "__main__":
    raise SystemExit(main())
