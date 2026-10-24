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
    with scheduled_backend.execution_context("transformer") as execution:
        scheduled = _model(scheduled_backend).forward([1, 2, 3])

    assert scheduled._values == pytest.approx(direct._values, abs=1e-12)
    assert scheduled_backend.stats.submitted > 0
    assert scheduled_backend.stats.completed == scheduled_backend.stats.submitted
    assert scheduled_backend.stats.failed == 0
    assert len(scheduled_backend.commands) == scheduled_backend.stats.submitted
    assert all(command.state == "completed" for command in scheduled_backend.commands)
    assert scheduled_backend.stats.planned_workgroups >= scheduled_backend.stats.submitted
    assert execution.stats.completed == execution.stats.submitted
    assert execution.stats.active is False
    assert execution.stats.elapsed_ns > 0
    assert execution.stats.command_ids[0].startswith("transformer-000001-")


def test_scheduled_backend_preserves_autodiff_for_vcpu_and_vgpu() -> None:
    for device in ("vcpu", "vgpu"):
        backend = ScheduledComputeBackend(ComputeRuntime(device), workgroup_size=1)
        model = _model(backend)
        with backend.execution_context("training") as execution:
            loss = model.forward([1, 2]).sum()
            loss.backward()

        assert all(parameter.grad is not None for parameter in model.parameters())
        assert backend.stats.completed == backend.stats.submitted
        assert backend.stats.failed == 0
        assert execution.stats.completed == execution.stats.submitted


def test_scheduled_vnpu_transformer_requires_frozen_parameters() -> None:
    backend = ScheduledComputeBackend(ComputeRuntime("vnpu"))
    model = _model(backend)

    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward([1, 2])

    model.requires_grad_(False)
    with backend.execution_context("inference") as execution:
        logits = model.forward([1, 2])
    assert logits.shape == (2, model.vocab_size)
    assert logits.requires_grad is False
    assert execution.stats.completed > 0


def test_scheduled_backend_enforces_memory_budget_and_records_failure() -> None:
    backend = ScheduledComputeBackend(
        ComputeRuntime("vcpu"),
        memory_budget_bytes=8,
    )

    with backend.execution_context("limited") as execution:
        with pytest.raises(MemoryError, match="reservation exceeds"):
            backend.matmul([[1.0, 2.0]], [[3.0], [4.0]])

    assert backend.stats.submitted == 0
    assert execution.stats.failed == 1
    assert execution.stats.peak_reserved_memory_bytes == 0


def test_scheduled_backend_history_is_bounded_and_clearable() -> None:
    backend = ScheduledComputeBackend(history_limit=2)
    for value in (1.0, 2.0, 3.0):
        backend.matmul([[value]], [[2.0]])

    assert len(backend.commands) == 2
    assert backend.stats.submitted == 3
    backend.clear_history()
    assert backend.commands == ()
    assert backend.stats.submitted == 0


def test_execution_context_cancellation_is_checked_between_kernels() -> None:
    backend = ScheduledComputeBackend()
    context = backend.execution_context("cancelled")
    context.cancel()

    with context:
        with pytest.raises(RuntimeError, match="was cancelled"):
            backend.matmul([[1.0]], [[2.0]])

    assert context.stats.cancelled == 1
    assert context.stats.submitted == 0
    assert context.stats.active is False


def test_execution_context_rejects_nesting_and_reentry() -> None:
    backend = ScheduledComputeBackend()
    first = backend.execution_context("outer")
    with first:
        with pytest.raises(RuntimeError, match="nested"):
            with backend.execution_context("inner"):
                pass
    with pytest.raises(RuntimeError, match="more than once"):
        with first:
            pass


def test_execution_context_cleans_up_after_model_failure() -> None:
    backend = ScheduledComputeBackend()
    model = _model(backend)
    with backend.execution_context("failure") as execution:
        with pytest.raises(ValueError, match="sequence exceeds"):
            model.forward([1, 2, 3, 4, 5, 6, 7])

    assert execution.stats.active is False
    assert backend._local.execution if hasattr(backend._local, "execution") else True
    with backend.execution_context("recovery") as recovered:
        model.forward([1, 2])
    assert recovered.stats.completed > 0


def test_execution_contexts_are_thread_local() -> None:
    from concurrent.futures import ThreadPoolExecutor

    backend = ScheduledComputeBackend()

    def run(value: float) -> tuple[float, str]:
        with backend.execution_context("parallel") as execution:
            output = backend.matmul([[value]], [[2.0]])
        return output[0][0], execution.execution_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, (2.0, 3.0)))

    assert sorted(value for value, _ in results) == [4.0, 6.0]
    assert len({execution_id for _, execution_id in results}) == 2
    assert backend.stats.completed == 2
