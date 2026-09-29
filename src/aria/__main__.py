"""CLI entry point for ARIA."""

from aria import __version__


def main() -> None:
    """Run the minimal ARIA entry point."""
    print(f"ARIA {__version__}")


if __name__ == "__main__":
    main()
