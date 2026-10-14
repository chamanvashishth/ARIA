"""Explicit dependency graphs for ARIA's software compute runtime.

The first graph engine intentionally supports matrix multiplication nodes only.
It executes on the selected host-CPU backend; worker threads provide scheduling
concurrency, not physical GPU/NPU acceleration.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal, Mapping, Sequence

from aria.compute.backends import ComputeRuntime, Matrix

GraphState = Literal["completed", "failed", "blocked"]


@dataclass(frozen=True)
class NodeRef:
    """A symbolic reference to the output of a graph node."""

    node_id: str


GraphInput = Matrix | NodeRef


@dataclass(frozen=True)
class GraphNode:
    """One immutable matrix operation and its symbolic inputs."""

    node_id: str
    operation: Literal["matmul"]
    left: GraphInput
    right: GraphInput

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(
            item.node_id
            for item in (self.left, self.right)
            if isinstance(item, NodeRef)
        )


@dataclass(frozen=True)
class GraphNodeMetric:
    """Execution outcome and estimated resource use for one graph node."""

    node_id: str
    state: GraphState
    elapsed_ns: int
    estimated_memory_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class GraphExecutionResult:
    """Successful node outputs plus per-node execution diagnostics."""

    outputs: Mapping[str, tuple[tuple[float, ...], ...]]
    node_metrics: tuple[GraphNodeMetric, ...]
    elapsed_ns: int

    def output(self, node: str | NodeRef) -> tuple[tuple[float, ...], ...]:
        """Return a node's immutable matrix result."""
        node_id = node.node_id if isinstance(node, NodeRef) else node
        try:
            return self.outputs[node_id]
        except KeyError as error:
            raise KeyError(f"no successful output for graph node: {node_id}") from error


class ComputeGraph:
    """Build and execute a validated directed acyclic matrix-operation graph.

    Graph references may point forward to nodes added later. Validation checks
    every reference and detects cycles before any computation starts.
    """

    def __init__(self, name: str = "compute-graph") -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("graph name must be a non-empty string")
        self.name = name
        self._nodes: dict[str, GraphNode] = {}

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        """Return nodes in insertion order."""
        return tuple(self._nodes.values())

    def matmul(self, node_id: str, left: GraphInput, right: GraphInput) -> NodeRef:
        """Add a matrix multiplication and return its symbolic output reference."""
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if node_id in self._nodes:
            raise ValueError(f"duplicate graph node: {node_id}")
        frozen_inputs: list[GraphInput] = []
        for operand_name, operand in (("left", left), ("right", right)):
            if isinstance(operand, NodeRef):
                if not operand.node_id:
                    raise ValueError(f"{operand_name} node reference must not be empty")
                frozen_inputs.append(operand)
                continue
            if (
                not isinstance(operand, Sequence)
                or isinstance(operand, (str, bytes))
                or not operand
            ):
                raise ValueError(f"{operand_name} operand must be a non-empty matrix")
            if any(
                not isinstance(row, Sequence) or isinstance(row, (str, bytes))
                for row in operand
            ):
                raise TypeError(f"{operand_name} matrix rows must be sequences")
            frozen_inputs.append(tuple(tuple(value for value in row) for row in operand))
        self._nodes[node_id] = GraphNode(
            node_id, "matmul", frozen_inputs[0], frozen_inputs[1]
        )
        return NodeRef(node_id)

    def validate(self) -> tuple[str, ...]:
        """Validate references and return a deterministic topological ordering."""
        for node in self._nodes.values():
            missing = [ref for ref in node.dependencies if ref not in self._nodes]
            if missing:
                raise ValueError(
                    f"node {node.node_id!r} references unknown nodes: {missing}"
                )

        temporary: set[str] = set()
        permanent: set[str] = set()
        ordered: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError(f"compute graph contains a cycle at node {node_id!r}")
            temporary.add(node_id)
            for dependency in self._nodes[node_id].dependencies:
                visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)
            ordered.append(node_id)

        for node_id in self._nodes:
            visit(node_id)
        return tuple(ordered)

    @staticmethod
    def _resolve(operand: GraphInput, outputs: Mapping[str, list[list[float]]]):
        if isinstance(operand, NodeRef):
            return outputs[operand.node_id]
        return operand

    @staticmethod
    def _estimate_memory(left: Matrix, right: Matrix) -> int:
        try:
            rows, inner = len(left), len(left[0])
            right_rows, columns = len(right), len(right[0])
            if inner != right_rows:
                return 0
            return 8 * (rows * inner + right_rows * columns + rows * columns)
        except (IndexError, TypeError):
            return 0

    def execute(
        self,
        runtime: ComputeRuntime | None = None,
        *,
        max_workers: int = 1,
    ) -> GraphExecutionResult:
        """Execute ready nodes in bounded parallel waves and retain node metrics.

        A failed node blocks only its dependents. Independent branches continue,
        and the result reports failures rather than discarding successful work.
        """
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        order = self.validate()
        selected_runtime = runtime or ComputeRuntime("vcpu")
        started = perf_counter_ns()
        outputs: dict[str, list[list[float]]] = {}
        states: dict[str, GraphState] = {}
        metrics: dict[str, GraphNodeMetric] = {}
        pending = set(order)

        def run_node(left: Matrix, right: Matrix):
            node_started = perf_counter_ns()
            try:
                value = selected_runtime.matmul(left, right)
                immutable = tuple(tuple(float(item) for item in row) for row in value)
                return immutable, perf_counter_ns() - node_started, None
            except Exception as error:
                return None, perf_counter_ns() - node_started, f"{type(error).__name__}: {error}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending:
                newly_blocked = [
                    node_id
                    for node_id in order
                    if node_id in pending
                    and any(
                        states.get(dep) in {"failed", "blocked"}
                        for dep in self._nodes[node_id].dependencies
                    )
                ]
                for node_id in newly_blocked:
                    states[node_id] = "blocked"
                    metrics[node_id] = GraphNodeMetric(
                        node_id=node_id,
                        state="blocked",
                        elapsed_ns=0,
                        estimated_memory_bytes=0,
                        error="one or more dependencies did not complete",
                    )
                    pending.remove(node_id)

                ready = [
                    node_id
                    for node_id in order
                    if node_id in pending
                    and all(
                        states.get(dep) == "completed"
                        for dep in self._nodes[node_id].dependencies
                    )
                ]
                if not ready:
                    if pending:
                        raise RuntimeError("validated graph could not make execution progress")
                    break

                futures = {}
                for node_id in ready:
                    node = self._nodes[node_id]
                    left = self._resolve(node.left, outputs)
                    right = self._resolve(node.right, outputs)
                    futures[executor.submit(run_node, left, right)] = (
                        node_id,
                        self._estimate_memory(left, right),
                    )
                    pending.remove(node_id)

                for future in as_completed(futures):
                    node_id, memory_bytes = futures[future]
                    value, elapsed_ns, error = future.result()
                    if error is None and value is not None:
                        outputs[node_id] = [list(row) for row in value]
                        states[node_id] = "completed"
                        metrics[node_id] = GraphNodeMetric(
                            node_id, "completed", elapsed_ns, memory_bytes
                        )
                    else:
                        states[node_id] = "failed"
                        metrics[node_id] = GraphNodeMetric(
                            node_id, "failed", elapsed_ns, memory_bytes, error
                        )

        return GraphExecutionResult(
            outputs={
                node_id: tuple(tuple(row) for row in value)
                for node_id, value in outputs.items()
            },
            node_metrics=tuple(metrics[node_id] for node_id in order),
            elapsed_ns=perf_counter_ns() - started,
        )
