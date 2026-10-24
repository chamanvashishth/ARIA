from pathlib import Path

from aria import __version__
from aria.config import AriaConfig


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_config_creates_local_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "aria-data"
    AriaConfig(data_dir=data_dir).ensure_directories()
    assert data_dir.is_dir()
