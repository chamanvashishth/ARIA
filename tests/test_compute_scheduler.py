import pytest

from aria.compute import ComputeRuntime, ScheduledComputeBackend
from aria.compute.scheduler import CommandQueue, ComputeScheduler


def test_queue_executes_independent_commands_in_submission_order() -> None:
    queue = CommandQueue(workgroup_size=1)
    first = queue.submit_matmul([[1, 2]], [[3], [4]])
    second = queue.submit_matmul([[2]], [[5]])

    snapshots = ComputeScheduler().run(queue)

    assert [item.command_id for item in snapshots] == [first, second]
    assert [item.state for item in snapshots] == ["completed", "completed"]
    assert queue.get(first).result == ((11.0,),)
    assert queue.get(second).result == ((10.0,),)


def test_dependencies_run_after_prerequisites() -> None:
    queue = CommandQueue()
    first = queue.submit_matmul([[2]], [[3]], command_id="prepare")
    queue.submit_matmul(
        [[4, 5]], [[6], [7]], depends_on=(first,), command_id="dependent"
    )

    ComputeScheduler().run(queue)

    assert queue.get("prepare").state == "completed"
    assert queue.get("dependent").state == "completed"
    assert queue.get("dependent").dependencies == ("prepare",)


def test_failed_command_blocks_dependents_but_not_independent_work() -> None:
    queue = CommandQueue()
    failed = queue.submit_matmul([[1]], [[2]], command_id="bad")
    dependent = queue.submit_matmul(
        [[2]], [[3]], depends_on=(failed,), command_id="blocked"
    )
    independent = queue.submit_matmul([[3]], [[4]], command_id="independent")

    # Inject malformed operand data to exercise backend-time failure handling.
    queue._commands[failed].left = ((object(),),)  # type: ignore[assignment]
    snapshots = ComputeScheduler().run(queue)

    assert queue.get(failed).state == "failed"
    assert queue.get(dependent).state == "blocked"
    assert queue.get(independent).state == "completed"
    assert snapshots[1].error is not None


def test_queue_rejects_unknown_and_duplicate_dependencies() -> None:
    queue = CommandQueue()
    with pytest.raises(ValueError, match="earlier commands"):
        queue.submit_matmul([[1]], [[1]], depends_on=("missing",))

    command = queue.submit_matmul([[1]], [[1]])
    with pytest.raises(ValueError, match="must be unique"):
        queue.submit_matmul([[1]], [[1]], depends_on=(command, command))


def test_queue_rejects_duplicate_and_self_command_ids() -> None:
    queue = CommandQueue()
    queue.submit_matmul([[1]], [[1]], command_id="fixed")
    with pytest.raises(ValueError, match="duplicate command_id"):
        queue.submit_matmul([[1]], [[1]], command_id="fixed")
    with pytest.raises(ValueError, match="cannot depend on itself"):
        queue.submit_matmul([[1]], [[1]], command_id="self", depends_on=("self",))


def test_queue_enforces_pending_capacity_and_releases_terminal_slots() -> None:
    queue = CommandQueue(max_pending_commands=1)
    queue.submit_matmul([[1]], [[2]])
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        queue.submit_matmul([[2]], [[3]])

    ComputeScheduler().run(queue)
    next_id = queue.submit_matmul([[2]], [[3]])
    assert next_id == "cmd-000002"


def test_memory_budget_is_reserved_and_released() -> None:
    queue = CommandQueue(memory_budget_bytes=24)
    first = queue.submit_matmul([[1]], [[2]])
    assert queue.reserved_memory_bytes == 24
    with pytest.raises(MemoryError, match="exceeds queue budget"):
        queue.submit_matmul([[1]], [[2]])

    ComputeScheduler().run(queue)
    assert queue.get(first).state == "completed"
    assert queue.reserved_memory_bytes == 0
    assert queue.peak_reserved_memory_bytes == 24


def test_workgroups_cover_output_matrix_without_overlap_or_gaps() -> None:
    queue = CommandQueue(workgroup_size=2)
    command_id = queue.submit_matmul([[1], [2], [3]], [[1, 2, 3, 4, 5]])
    groups = queue.get(command_id).workgroups

    assert len(groups) == 6
    covered = {
        (row, column)
        for group in groups
        for row in range(group.row_start, group.row_stop)
        for column in range(group.column_start, group.column_stop)
    }
    assert covered == {(row, column) for row in range(3) for column in range(5)}
    assert len(covered) == 15


def test_scheduler_stats_track_states_memory_and_workgroups() -> None:
    queue = CommandQueue(workgroup_size=1)
    queue.submit_matmul([[1, 2]], [[3], [4]])
    queue.submit_matmul([[2]], [[5]])

    ComputeScheduler().run(queue)
    stats = ComputeScheduler.stats(queue)

    assert stats.submitted == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.blocked == 0
    assert stats.queued == 0
    assert stats.running == 0
    assert stats.reserved_memory_bytes == 0
    assert stats.peak_reserved_memory_bytes > 0
    assert stats.planned_workgroups == 2


def test_scheduler_uses_selected_compute_runtime() -> None:
    queue = CommandQueue()
    command_id = queue.submit_matmul([[1.25, -2.0]], [[2.0], [3.0]])

    ComputeScheduler(ComputeRuntime("vgpu")).run(queue)

    assert queue.get(command_id).result == ((-3.5,),)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [[1]], "left operand must be a non-empty matrix"),
        ([[1]], [], "right operand must be a non-empty matrix"),
        ([[1, 2]], [[1]], "matrix dimensions do not match"),
        ([[1], [2, 3]], [[1]], "left operand must be rectangular"),
    ],
)
def test_queue_validates_matrix_shapes(left, right, message) -> None:
    with pytest.raises(ValueError, match=message):
        CommandQueue().submit_matmul(left, right)


def test_scheduled_backend_batches_independent_products() -> None:
    backend = ScheduledComputeBackend(workgroup_size=1)
    results = backend.matmul_batch(
        (
            ([[1.0, 2.0]], [[3.0], [4.0]]),
            ([[2.0]], [[5.0]]),
            ([[1.0], [2.0]], [[2.0, 0.0]]),
        )
    )

    assert results == ([[11.0]], [[10.0]], [[2.0, 0.0], [4.0, 0.0]])
    assert backend.stats.submitted == 3
    assert backend.stats.completed == 3
    assert len(backend.commands) == 3


def test_scheduled_backend_batch_reserves_memory_for_all_operations() -> None:
    backend = ScheduledComputeBackend(memory_budget_bytes=40)
    with pytest.raises(MemoryError, match="exceeds queue budget"):
        backend.matmul_batch(
            (
                ([[1.0, 2.0]], [[3.0], [4.0]]),
                ([[1.0, 2.0]], [[3.0], [4.0]]),
            )
        )

    assert backend.stats.submitted == 0
    assert backend.stats.failed == 1


def test_scheduled_backend_batch_validates_dependency_indices() -> None:
    backend = ScheduledComputeBackend()
    with pytest.raises(ValueError, match="earlier batch indices"):
        backend.matmul_batch(
            (
                ([[1.0]], [[2.0]]),
                ([[2.0]], [[3.0]]),
            ),
            dependencies=((), (1,)),
        )
    assert backend.stats.submitted == 0


def test_scheduled_backend_batch_executes_dependencies_in_order() -> None:
    backend = ScheduledComputeBackend()
    results = backend.matmul_batch(
        (
            ([[1.0]], [[2.0]]),
            ([[3.0]], [[4.0]]),
            ([[5.0]], [[6.0]]),
        ),
        dependencies=((), (0,), (1,)),
    )

    assert results == ([[2.0]], [[12.0]], [[30.0]])
    assert backend.commands[1].dependencies == (backend.commands[0].command_id,)
    assert backend.commands[2].dependencies == (backend.commands[1].command_id,)


def test_scheduled_backend_batch_propagates_runtime_failures() -> None:
    class FailingRuntime:
        def __init__(self) -> None:
            self._runtime = ComputeRuntime("vcpu")
            self.info = self._runtime.info

        def matmul(self, left, right):
            if left[0][0] == 2.0:
                raise ArithmeticError("deliberate test failure")
            return self._runtime.matmul(left, right)

    backend = ScheduledComputeBackend(runtime=FailingRuntime())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ended in state failed"):
        backend.matmul_batch(
            (
                ([[1.0]], [[2.0]]),
                ([[2.0]], [[3.0]]),
                ([[4.0]], [[5.0]]),
            ),
            dependencies=((), (0,), (1,)),
        )

    assert [command.state for command in backend.commands] == [
        "completed",
        "failed",
        "blocked",
    ]
    assert backend.stats.completed == 1
    assert backend.stats.failed == 2


def test_execution_context_tracks_whole_batch_memory_and_commands() -> None:
    backend = ScheduledComputeBackend(workgroup_size=1)
    with backend.execution_context("batch") as execution:
        results = backend.matmul_batch(
            (
                ([[1.0, 2.0]], [[3.0], [4.0]]),
                ([[2.0, 3.0]], [[4.0], [5.0]]),
            )
        )

    assert results == ([[11.0]], [[23.0]])
    assert execution.stats.submitted == 2
    assert execution.stats.completed == 2
    assert execution.stats.peak_reserved_memory_bytes == 80
    assert len(execution.stats.command_ids) == 2
