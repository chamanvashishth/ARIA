"""Smoke test for the ARIA CLI entry point."""

import sys
from subprocess import run


def test_cli_starts() -> None:
    result = run(
        [sys.executable, "-m", "aria"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ARIA 0.1.0"
