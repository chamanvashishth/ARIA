"""Deterministic command queues and workgroup planning for ARIA compute.

This is a synchronous host-CPU orchestration layer. Workgroups are logical
output-tile descriptors, not independent hardware dispatches.
"""
from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Event, Lock, local
from time import perf_counter_ns
from typing import Literal, Sequence

from aria.compute.backends import ComputeRuntime, Matrix

CommandState = Literal["queued", "running", "completed", "failed", "blocked"]
_TERMINAL = frozenset({"completed", "failed", "blocked"})


@dataclass(frozen=True)
class Workgroup:
    workgroup_id: str
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int


@dataclass(frozen=True)
class ComputeCommand:
    command_id: str
    operation: str
    dependencies: tuple[str, ...]
    state: CommandState
    workgroups: tuple[Workgroup, ...]
    reserved_memory_bytes: int
    submitted_at_ns: int
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    result: tuple[tuple[float, ...], ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class SchedulerStats:
    submitted: int
    queued: int
    running: int
    completed: int
    failed: int
    blocked: int
    reserved_memory_bytes: int
    peak_reserved_memory_bytes: int
    planned_workgroups: int


@dataclass
class _CommandRecord:
    command_id: str
    left: tuple[tuple[int | float, ...], ...]
    right: tuple[tuple[int | float, ...], ...]
    dependencies: tuple[str, ...]
    workgroups: tuple[Workgroup, ...]
    reserved_memory_bytes: int
    submitted_at_ns: int
    state: CommandState = "queued"
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    result: tuple[tuple[float, ...], ...] | None = None
    error: str | None = None

    def snapshot(self) -> ComputeCommand:
        return ComputeCommand(
            command_id=self.command_id,
            operation="matmul",
            dependencies=self.dependencies,
            state=self.state,
            workgroups=self.workgroups,
            reserved_memory_bytes=self.reserved_memory_bytes,
            submitted_at_ns=self.submitted_at_ns,
            started_at_ns=self.started_at_ns,
            completed_at_ns=self.completed_at_ns,
            result=self.result,
            error=self.error,
        )


def _dimensions(matrix: Matrix, name: str) -> tuple[int, int]:
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)) or not matrix:
        raise ValueError(f"{name} must be a non-empty matrix")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in matrix):
        raise TypeError(f"{name} rows must be sequences")
    columns = len(matrix[0])
    if columns == 0 or any(len(row) != columns for row in matrix):
        raise ValueError(f"{name} must be rectangular and non-empty")
    return len(matrix), columns


def _plan_workgroups(
    command_id: str, rows: int, columns: int, tile_size: int
) -> tuple[Workgroup, ...]:
    groups: list[Workgroup] = []
    for row_start in range(0, rows, tile_size):
        for column_start in range(0, columns, tile_size):
            ordinal = len(groups)
            groups.append(
                Workgroup(
                    workgroup_id=f"{command_id}:wg-{ordinal:04d}",
                    row_start=row_start,
                    row_stop=min(row_start + tile_size, rows),
                    column_start=column_start,
                    column_stop=min(column_start + tile_size, columns),
                )
            )
    return tuple(groups)


class CommandQueue:
    """FIFO queue with earlier-command dependencies and resource reservations."""

    def __init__(
        self,
        *,
        max_pending_commands: int = 128,
        memory_budget_bytes: int | None = None,
        workgroup_size: int = 16,
    ) -> None:
        if isinstance(max_pending_commands, bool) or not isinstance(max_pending_commands, int) or max_pending_commands <= 0:
            raise ValueError("max_pending_commands must be a positive integer")
        if memory_budget_bytes is not None and (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ValueError("memory_budget_bytes must be a positive integer or None")
        if isinstance(workgroup_size, bool) or not isinstance(workgroup_size, int) or workgroup_size <= 0:
            raise ValueError("workgroup_size must be a positive integer")

        self.max_pending_commands = max_pending_commands
        self.memory_budget_bytes = memory_budget_bytes
        self.workgroup_size = workgroup_size
        self._commands: dict[str, _CommandRecord] = {}
        self._next_id = 1
        self._reserved_memory_bytes = 0
        self._peak_reserved_memory_bytes = 0

    @property
    def commands(self) -> tuple[ComputeCommand, ...]:
        return tuple(record.snapshot() for record in self._commands.values())

    @property
    def reserved_memory_bytes(self) -> int:
        return self._reserved_memory_bytes

    @property
    def peak_reserved_memory_bytes(self) -> int:
        return self._peak_reserved_memory_bytes

    def get(self, command_id: str) -> ComputeCommand:
        try:
            return self._commands[command_id].snapshot()
        except KeyError as error:
            raise KeyError(f"unknown command: {command_id}") from error

    def submit_matmul(
        self,
        left: Matrix,
        right: Matrix,
        *,
        depends_on: Sequence[str] = (),
        command_id: str | None = None,
    ) -> str:
        left_rows, inner = _dimensions(left, "left operand")
        right_rows, columns = _dimensions(right, "right operand")
        if inner != right_rows:
            raise ValueError(
                f"matrix dimensions do not match: {left_rows}x{inner} cannot multiply {right_rows}x{columns}"
            )

        active = sum(record.state not in _TERMINAL for record in self._commands.values())
        if active >= self.max_pending_commands:
            raise RuntimeError("command queue capacity exceeded")

        dependencies = tuple(depends_on)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("command dependencies must be unique")
        if command_id is None:
            while f"cmd-{self._next_id:06d}" in self._commands:
                self._next_id += 1
            selected_id = f"cmd-{self._next_id:06d}"
            self._next_id += 1
        else:
            if not isinstance(command_id, str) or not command_id:
                raise ValueError("command_id must be a non-empty string")
            selected_id = command_id
        if selected_id in self._commands:
            raise ValueError(f"duplicate command_id: {selected_id}")
        if selected_id in dependencies:
            raise ValueError("a command cannot depend on itself")
        missing = [item for item in dependencies if item not in self._commands]
        if missing:
            raise ValueError(f"dependencies must reference earlier commands: {missing}")

        memory_bytes = 8 * (left_rows * inner + right_rows * columns + left_rows * columns)
        requested = self._reserved_memory_bytes + memory_bytes
        if self.memory_budget_bytes is not None and requested > self.memory_budget_bytes:
            raise MemoryError(
                f"command memory reservation exceeds queue budget: {requested} > {self.memory_budget_bytes} bytes"
            )

        left_copy = tuple(tuple(value for value in row) for row in left)
        right_copy = tuple(tuple(value for value in row) for row in right)
        record = _CommandRecord(
            command_id=selected_id,
            left=left_copy,
            right=right_copy,
            dependencies=dependencies,
            workgroups=_plan_workgroups(
                selected_id, left_rows, columns, self.workgroup_size
            ),
            reserved_memory_bytes=memory_bytes,
            submitted_at_ns=perf_counter_ns(),
        )
        self._commands[selected_id] = record
        self._reserved_memory_bytes = requested
        self._peak_reserved_memory_bytes = max(self._peak_reserved_memory_bytes, requested)
        return selected_id

    def _release(self, record: _CommandRecord) -> None:
        self._reserved_memory_bytes -= record.reserved_memory_bytes
        if self._reserved_memory_bytes < 0:
            raise RuntimeError("internal scheduler memory accounting underflow")


class ComputeScheduler:
    """Execute ready commands synchronously in deterministic submission order."""

    def __init__(self, runtime: ComputeRuntime | None = None) -> None:
        self.runtime = runtime or ComputeRuntime()

    def run(self, queue: CommandQueue) -> tuple[ComputeCommand, ...]:
        for record in queue._commands.values():
            if record.state != "queued":
                continue
            dependencies = [queue._commands[item] for item in record.dependencies]
            failed = next(
                (item for item in dependencies if item.state in {"failed", "blocked"}),
                None,
            )
            if failed is not None:
                record.state = "blocked"
                record.error = f"dependency {failed.command_id} ended in state {failed.state}"
                record.completed_at_ns = perf_counter_ns()
                queue._release(record)
                continue
            if any(item.state != "completed" for item in dependencies):
                continue

            record.state = "running"
            record.started_at_ns = perf_counter_ns()
            try:
                output = self.runtime.matmul(record.left, record.right)
                record.result = tuple(tuple(value for value in row) for row in output)
                record.state = "completed"
            except Exception as error:
                record.state = "failed"
                record.error = f"{type(error).__name__}: {error}"
            finally:
                record.completed_at_ns = perf_counter_ns()
                queue._release(record)

        return queue.commands

    @staticmethod
    def stats(queue: CommandQueue) -> SchedulerStats:
        states = [record.state for record in queue._commands.values()]
        return SchedulerStats(
            submitted=len(states),
            queued=states.count("queued"),
            running=states.count("running"),
            completed=states.count("completed"),
            failed=states.count("failed"),
            blocked=states.count("blocked"),
            reserved_memory_bytes=queue.reserved_memory_bytes,
            peak_reserved_memory_bytes=queue.peak_reserved_memory_bytes,
            planned_workgroups=sum(len(item.workgroups) for item in queue._commands.values()),
        )


@dataclass(frozen=True)
class ScheduledExecutionStats:
    """Immutable aggregate measurements for one model execution scope."""

    execution_id: str
    submitted: int
    completed: int
    failed: int
    cancelled: int
    elapsed_ns: int
    peak_reserved_memory_bytes: int
    planned_workgroups: int
    command_ids: tuple[str, ...]
    active: bool


class ScheduledExecutionContext(AbstractContextManager["ScheduledExecutionContext"]):
    """Own one model call's lifecycle, cancellation signal and resource summary.

    Cancellation is cooperative and checked between synchronous matrix kernels.
    It cannot interrupt a Python matrix multiplication that is already running.
    """

    def __init__(self, backend: "ScheduledComputeBackend", execution_id: str) -> None:
        self.backend = backend
        self.execution_id = execution_id
        self._cancel_event = Event()
        self._entered = False
        self._active = False
        self._started_at_ns: int | None = None
        self._finished_at_ns: int | None = None
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._peak_reserved_memory_bytes = 0
        self._planned_workgroups = 0
        self._command_ids: list[str] = []

    def __enter__(self) -> "ScheduledExecutionContext":
        if self._entered:
            raise RuntimeError("execution context cannot be entered more than once")
        if getattr(self.backend._local, "execution", None) is not None:
            raise RuntimeError("nested scheduled execution contexts are not supported")
        self._entered = True
        self._active = True
        self._started_at_ns = perf_counter_ns()
        self.backend._local.execution = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._active = False
        self._finished_at_ns = perf_counter_ns()
        if getattr(self.backend._local, "execution", None) is self:
            del self.backend._local.execution
        return False

    def cancel(self) -> None:
        """Request cancellation before the next scheduled matrix operation."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def stats(self) -> ScheduledExecutionStats:
        end = self._finished_at_ns
        if end is None:
            end = perf_counter_ns()
        start = self._started_at_ns or end
        return ScheduledExecutionStats(
            execution_id=self.execution_id,
            submitted=self._submitted,
            completed=self._completed,
            failed=self._failed,
            cancelled=self._cancelled,
            elapsed_ns=max(0, end - start),
            peak_reserved_memory_bytes=self._peak_reserved_memory_bytes,
            planned_workgroups=self._planned_workgroups,
            command_ids=tuple(self._command_ids),
            active=self._active,
        )

    def _record_submitted(self, command_id: str) -> None:
        self._submitted += 1
        self._command_ids.append(command_id)

    def _record_command(self, command: ComputeCommand) -> None:
        self._planned_workgroups += len(command.workgroups)
        self._peak_reserved_memory_bytes = max(
            self._peak_reserved_memory_bytes, command.reserved_memory_bytes
        )
        if command.state == "completed":
            self._completed += 1
        else:
            self._failed += 1

    def _record_failure(self) -> None:
        self._failed += 1

    def _record_cancelled(self) -> None:
        self._cancelled += 1


class ScheduledComputeBackend:
    """Synchronous scheduler adapter implementing the neural ComputeBackend protocol.

    Use execution_context() around a model forward/generation call to collect
    per-execution counters, timings and estimated memory use. Cancellation is
    checked between kernels. Matrix kernels remain synchronous and cannot be
    interrupted once running.
    """

    def __init__(
        self,
        runtime: ComputeRuntime | None = None,
        *,
        workgroup_size: int = 16,
        memory_budget_bytes: int | None = None,
        history_limit: int = 1024,
    ) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit <= 0:
            raise ValueError("history_limit must be a positive integer")
        self.runtime = runtime or ComputeRuntime()
        self.workgroup_size = workgroup_size
        self.memory_budget_bytes = memory_budget_bytes
        self.history_limit = history_limit
        self._history: deque[ComputeCommand] = deque(maxlen=history_limit)
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._peak_reserved_memory_bytes = 0
        self._planned_workgroups = 0
        self._lock = Lock()
        self._local = local()
        self._next_execution_id = 1
        self._next_command_id = 1

    @property
    def info(self):
        """Expose the wrapped backend's capabilities unchanged."""
        return self.runtime.info

    @property
    def commands(self) -> tuple[ComputeCommand, ...]:
        """Return the most recent immutable scheduler command snapshots."""
        with self._lock:
            return tuple(self._history)

    @property
    def stats(self) -> SchedulerStats:
        with self._lock:
            return SchedulerStats(
                submitted=self._submitted,
                queued=0,
                running=0,
                completed=self._completed,
                failed=self._failed,
                blocked=0,
                reserved_memory_bytes=0,
                peak_reserved_memory_bytes=self._peak_reserved_memory_bytes,
                planned_workgroups=self._planned_workgroups,
            )

    def execution_context(self, label: str | None = None) -> ScheduledExecutionContext:
        """Create an execution scope; label becomes a readable execution ID prefix."""
        if label is not None and (
            not isinstance(label, str) or not label.strip() or any(ch.isspace() for ch in label)
        ):
            raise ValueError("execution label must be a non-empty token without whitespace")
        with self._lock:
            ordinal = self._next_execution_id
            self._next_execution_id += 1
        prefix = label or "execution"
        return ScheduledExecutionContext(self, f"{prefix}-{ordinal:06d}")

    def clear_history(self) -> None:
        """Clear retained command snapshots and reset aggregate counters."""
        with self._lock:
            self._history.clear()
            self._submitted = 0
            self._completed = 0
            self._failed = 0
            self._peak_reserved_memory_bytes = 0
            self._planned_workgroups = 0

    def matmul(self, left: Matrix, right: Matrix) -> list[list[float]]:
        context = getattr(self._local, "execution", None)
        if context is not None and context.cancelled:
            context._record_cancelled()
            raise RuntimeError(f"scheduled execution {context.execution_id} was cancelled")

        with self._lock:
            ordinal = self._next_command_id
            self._next_command_id += 1
        execution_prefix = context.execution_id if context is not None else "model"
        command_id = f"{execution_prefix}-matmul-{ordinal:08d}"
        queue = CommandQueue(
            max_pending_commands=1,
            memory_budget_bytes=self.memory_budget_bytes,
            workgroup_size=self.workgroup_size,
        )
        try:
            queue.submit_matmul(left, right, command_id=command_id)
        except Exception:
            if context is not None:
                context._record_failure()
            with self._lock:
                self._failed += 1
            raise

        if context is not None:
            context._record_submitted(command_id)
        with self._lock:
            self._submitted += 1

        command = ComputeScheduler(self.runtime).run(queue)[0]
        with self._lock:
            self._history.append(command)
            self._planned_workgroups += len(command.workgroups)
            self._peak_reserved_memory_bytes = max(
                self._peak_reserved_memory_bytes, command.reserved_memory_bytes
            )
            if command.state == "completed" and command.result is not None:
                self._completed += 1
            else:
                self._failed += 1
        if context is not None:
            context._record_command(command)

        if command.state != "completed" or command.result is None:
            raise RuntimeError(
                f"scheduled matrix multiplication {command.command_id} "
                f"ended in state {command.state}: {command.error or 'no result'}"
            )
        return [list(row) for row in command.result]
