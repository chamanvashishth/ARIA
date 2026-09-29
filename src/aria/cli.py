"""Command-line entry point for the ARIA foundation."""

from aria import __version__
from aria.config import AriaConfig
from aria.logging import configure_logging


def main() -> None:
    config = AriaConfig()
    config.ensure_directories()
    configure_logging(config.log_level)
    print(f"ARIA {__version__} — engineering foundation ready")


if __name__ == "__main__":
    main()
