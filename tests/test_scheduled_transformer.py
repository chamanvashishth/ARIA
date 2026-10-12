import pytest

from aria.brain import TransformerLanguageModel
from aria.compute import (
    ComputeRuntime,
    ScheduledComputeBackend,
    VirtualCPUBackend,
)


def _model(backend):
    return TransformerLanguageModel(
        vocab_size=11,
        hidden_size=4,
        intermediate_size=7,
        num_layers=1,
        max_sequence_length=6,
        seed=91,
        backend=backend,
    )


def test_scheduled_backend_matches_direct_transformer_and_records_commands() -> None:
    direct = _model(VirtualCPUBackend()).forward([1, 2, 3])
    scheduled_backend = ScheduledComputeBackend(
        ComputeRuntime("vcpu"),
        workgroup_size=2,
        history_limit=128,
    )
    scheduled = _model(scheduled_backend).forward([1, 2, 3])

    assert scheduled._values == pytest.approx(direct._values, abs=1e-12)
    assert scheduled_backend.stats.submitted > 0
    assert scheduled_backend.stats.completed == scheduled_backend.stats.submitted
    assert scheduled_backend.stats.failed == 0
    assert len(scheduled_backend.commands) == scheduled_backend.stats.submitted
    assert all(command.state == "completed" for command in scheduled_backend.commands)
    assert scheduled_backend.stats.planned_workgroups >= scheduled_backend.stats.submitted


def test_scheduled_backend_preserves_autodiff_for_vcpu_and_vgpu() -> None:
    from aria.brain import Parameter

    for device in ("vcpu", "vgpu"):
        backend = ScheduledComputeBackend(ComputeRuntime(device), workgroup_size=1)
        model = _model(backend)
        loss = model.forward([1, 2]).sum()
        loss.backward()

        assert all(parameter.grad is not None for parameter in model.parameters())
        assert backend.stats.completed == backend.stats.submitted
        assert backend.stats.failed == 0


def test_scheduled_vnpu_transformer_requires_frozen_parameters() -> None:
    backend = ScheduledComputeBackend(ComputeRuntime("vnpu"))
    model = _model(backend)

    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward([1, 2])

    model.requires_grad_(False)
    logits = model.forward([1, 2])
    assert logits.shape == (2, model.vocab_size)
    assert logits.requires_grad is False
    assert backend.stats.completed > 0


def test_scheduled_backend_enforces_memory_budget_and_records_failure() -> None:
    backend = ScheduledComputeBackend(
        ComputeRuntime("vcpu"),
        memory_budget_bytes=8,
    )

    with pytest.raises(MemoryError, match="reservation exceeds"):
        backend.matmul([[1.0, 2.0]], [[3.0], [4.0]])

    assert backend.stats.submitted == 0


def test_scheduled_backend_history_is_bounded_and_clearable() -> None:
    backend = ScheduledComputeBackend(history_limit=2)
    for value in (1.0, 2.0, 3.0):
        backend.matmul([[value]], [[2.0]])

    assert len(backend.commands) == 2
    assert backend.stats.submitted == 3
    backend.clear_history()
    assert backend.commands == ()
    assert backend.stats.submitted == 0
