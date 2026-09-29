from aria.core.contracts import Capability, SystemStatus
from aria.hardware.models import HardwareProfile


def test_capability_rejects_unknown_status() -> None:
    try:
        Capability(name="demo", status="DONE")
    except ValueError:
        return
    raise AssertionError("Unknown capability status must be rejected")


def test_hardware_profile_is_serializable() -> None:
    profile = HardwareProfile(
        operating_system="test",
        cpu_name="test-cpu",
        cpu_architecture="x86_64",
        cpu_cores=8,
        cpu_threads=16,
        system_ram_bytes=16 * 1024**3,
        gpu_available=False,
        gpu_name=None,
        gpu_vram_bytes=None,
        gpu_backend=None,
        npu_available=False,
        npu_name=None,
        npu_runtime=None,
        storage_free_bytes=None,
    )
    assert profile.to_dict()["cpu_threads"] == 16


def test_system_status_marks_foundation_explicitly() -> None:
    status = SystemStatus("ARIA", "0.1.0", "foundation", False)
    assert status.implemented is False
