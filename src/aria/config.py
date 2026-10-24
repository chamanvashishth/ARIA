"""Application configuration for ARIA."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AriaConfig:
    """Minimal configuration boundary shared by ARIA components."""

    data_dir: Path = Path("data")
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
