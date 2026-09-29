from aria.core.contracts import Capability, SystemStatus


def test_capability_rejects_unknown_status() -> None:
    try:
        Capability(name="demo", status="DONE")
    except ValueError:
        return
    raise AssertionError("Unknown capability status must be rejected")


def test_system_status_marks_foundation_explicitly() -> None:
    status = SystemStatus("ARIA", "0.1.0", "foundation", False)
    assert status.implemented is False
