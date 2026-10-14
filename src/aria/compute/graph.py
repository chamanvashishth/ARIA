"""Explicit dependency graphs for ARIA's software compute runtime.

Graph operations execute on the host CPU. Matrix multiplication may use the
selected software backend; elementwise operations use Python implementations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import fsum
from time import perf_counter_ns
from typing import Literal, Mapping, cast

from aria.compute.backends import ComputeRuntime, Matrix

GraphState = Literal["completed", "failed", "blocked"]
GraphValue = float | int | tuple["GraphValue", ...] | list["GraphValue"]
GraphOperation = Literal["matmul", "add", "multiply", "relu", "sum", "reshape", "bias_add"]


@dataclass(frozen=True)
class NodeRef:
    """A symbolic reference to the output of a graph node."""

    node_id: str


GraphInput = GraphValue | NodeRef


@dataclass(frozen=True)
class GraphNode:
    """One immutable operation and its symbolic or literal inputs."""

    node_id: str
    operation: GraphOperation
    inputs: tuple[GraphInput, ...]
    shape: tuple[int, ...] | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(item.node_id for item in self.inputs if isinstance(item, NodeRef))

    @property
    def left(self) -> GraphInput:
        """Compatibility accessor for the first operand."""
        return self.inputs[0]

    @property
    def right(self) -> GraphInput:
        """Compatibility accessor for the second operand."""
        if len(self.inputs) < 2:
            raise AttributeError("unary graph nodes do not have a right operand")
        return self.inputs[1]


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

    outputs: Mapping[str, object]
    node_metrics: tuple[GraphNodeMetric, ...]
    elapsed_ns: int

    def output(self, node: str | NodeRef) -> object:
        """Return a node's immutable result."""
        node_id = node.node_id if isinstance(node, NodeRef) else node
        try:
            return self.outputs[node_id]
        except KeyError as error:
            raise KeyError(f"no successful output for graph node: {node_id}") from error


def _freeze(value: object) -> GraphValue:
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError("graph values must contain only numeric scalars")
    if isinstance(value, (int, float)):
        return value
    return tuple(_freeze(item) for item in value)


def _shape(value: GraphValue) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("graph tensors must be rectangular")
    return (len(value),) + child


def _map_binary(left: GraphValue, right: GraphValue, operation: str) -> GraphValue:
    if _shape(left) != _shape(right):
        raise ValueError(f"shape mismatch: {_shape(left)} != {_shape(right)}")
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return tuple(_map_binary(a, b, operation) for a, b in zip(left, right))
    a, b = float(cast(float | int, left)), float(cast(float | int, right))
    return a + b if operation == "add" else a * b


def _map_relu(value: GraphValue) -> GraphValue:
    if isinstance(value, (tuple, list)):
        return tuple(_map_relu(item) for item in value)
    return max(0.0, float(value))


def _reshape_value(value: GraphValue, shape: tuple[int, ...]) -> GraphValue:
    flat: list[float | int] = []

    def collect(item: GraphValue) -> None:
        if isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        else:
            flat.append(item)

    collect(value)
    if any(dimension < 0 for dimension in shape):
        raise ValueError("reshape dimensions must be non-negative")
    size = 1
    for dimension in shape:
        size *= dimension
    if size != len(flat):
        raise ValueError("reshape must preserve the number of values")

    def build(offset: int, dimensions: tuple[int, ...]) -> tuple[GraphValue, int]:
        if not dimensions:
            return flat[offset], offset + 1
        items: list[GraphValue] = []
        for _ in range(dimensions[0]):
            item, offset = build(offset, dimensions[1:])
            items.append(item)
        return tuple(items), offset

    if not shape:
        return flat[0]
    return build(0, shape)[0]


def _bias_add(matrix: GraphValue, bias: GraphValue) -> GraphValue:
    if len(_shape(matrix)) != 2 or len(_shape(bias)) != 1:
        raise ValueError("bias_add requires a 2-D matrix and a 1-D bias")
    rows = cast(tuple[GraphValue, ...], matrix)
    bias_values = cast(tuple[GraphValue, ...] | list[GraphValue], bias)
    if any(len(cast(tuple[GraphValue, ...], row)) != len(bias_values) for row in rows):
        raise ValueError("bias width must match matrix columns")
    return tuple(
        tuple(float(value) + float(bias_values[column]) for column, value in enumerate(cast(tuple[GraphValue, ...], row)))
        for row in rows
    )


def _sum_values(value: GraphValue) -> float:
    if isinstance(value, (tuple, list)):
        return fsum(_sum_values(item) for item in value)
    return float(value)


class ComputeGraph:
    """Build and execute a validated directed acyclic operation graph.

    Supports matmul, shape-matched add/multiply, ReLU, and sum. References may
    point forward to nodes added later. Failed nodes block only their dependents.
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

    def _add(self, node_id: str, operation: GraphOperation, *inputs: object) -> NodeRef:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if node_id in self._nodes:
            raise ValueError(f"duplicate graph node: {node_id}")
        expected = 1 if operation in {"relu", "sum"} else 2
        if len(inputs) != expected:
            raise ValueError(f"{operation} requires {expected} input(s)")
        frozen: list[GraphInput] = []
        for value in inputs:
            if isinstance(value, NodeRef):
                if not value.node_id:
                    raise ValueError("node reference must not be empty")
                frozen.append(value)
            else:
                frozen.append(_freeze(value))
        self._nodes[node_id] = GraphNode(node_id, operation, tuple(frozen))
        return NodeRef(node_id)

    def matmul(self, node_id: str, left: object, right: object) -> NodeRef:
        """Add a rank-two matrix multiplication."""
        return self._add(node_id, "matmul", left, right)

    def add(self, node_id: str, left: object, right: object) -> NodeRef:
        """Add two values with identical shapes."""
        return self._add(node_id, "add", left, right)

    def multiply(self, node_id: str, left: object, right: object) -> NodeRef:
        """Multiply two values elementwise with identical shapes."""
        return self._add(node_id, "multiply", left, right)

    def relu(self, node_id: str, value: object) -> NodeRef:
        """Apply ReLU elementwise."""
        return self._add(node_id, "relu", value)

    def sum(self, node_id: str, value: object) -> NodeRef:
        """Reduce all elements to one scalar."""
        return self._add(node_id, "sum", value)

    def reshape(self, node_id: str, value: object, shape: tuple[int, ...]) -> NodeRef:
        """Reshape a value without changing its number of elements."""
        reference = self._add(node_id, "reshape", value)
        node = self._nodes[node_id]
        self._nodes[node_id] = GraphNode(node.node_id, node.operation, node.inputs, tuple(shape))
        return reference

    def bias_add(self, node_id: str, matrix: object, bias: object) -> NodeRef:
        """Add a feature bias vector to every row of a 2-D matrix."""
        return self._add(node_id, "bias_add", matrix, bias)

    def validate(self) -> tuple[str, ...]:
        """Validate references and return a deterministic topological ordering."""
        for node in self._nodes.values():
            missing = [ref for ref in node.dependencies if ref not in self._nodes]
            if missing:
                raise ValueError(f"node {node.node_id!r} references unknown nodes: {missing}")

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
    def _resolve(value: GraphInput, outputs: Mapping[str, object]) -> GraphValue:
        resolved = outputs[value.node_id] if isinstance(value, NodeRef) else value
        return cast(GraphValue, resolved)

    @staticmethod
    def _estimate_memory(values: tuple[GraphValue, ...], output: GraphValue | None = None) -> int:
        def count(value: GraphValue) -> int:
            if isinstance(value, (tuple, list)):
                return int(fsum(count(item) for item in value))
            return 1
        return 8 * (sum(count(value) for value in values) + (count(output) if output is not None else 0))

    def execute(
        self,
        runtime: ComputeRuntime | None = None,
        *,
        max_workers: int = 1,
    ) -> GraphExecutionResult:
        """Execute independent ready nodes in bounded parallel waves."""
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        order = self.validate()
        selected_runtime = runtime or ComputeRuntime("vcpu")
        started = perf_counter_ns()
        outputs: dict[str, object] = {}
        states: dict[str, GraphState] = {}
        metrics: dict[str, GraphNodeMetric] = {}
        pending = set(order)

        def run_node(node: GraphNode, values: tuple[GraphValue, ...]):
            node_started = perf_counter_ns()
            try:
                if node.operation == "matmul":
                    left, right = values
                    if len(_shape(left)) != 2 or len(_shape(right)) != 2:
                        raise ValueError("matmul operands must both be 2-D matrices")
                    result: GraphValue = tuple(
                        tuple(float(item) for item in row)
                        for row in selected_runtime.matmul(
                            cast(Matrix, left), cast(Matrix, right)
                        )
                    )
                elif node.operation in {"add", "multiply"}:
                    result = _map_binary(values[0], values[1], node.operation)
                elif node.operation == "relu":
                    result = _map_relu(values[0])
                elif node.operation == "sum":
                    result = _sum_values(values[0])
                elif node.operation == "reshape":
                    if node.shape is None:
                        raise ValueError("reshape node is missing its target shape")
                    result = _reshape_value(values[0], node.shape)
                else:
                    result = _bias_add(values[0], values[1])
                return result, perf_counter_ns() - node_started, None
            except Exception as error:
                return None, perf_counter_ns() - node_started, f"{type(error).__name__}: {error}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending:
                blocked = [
                    node_id for node_id in order
                    if node_id in pending and any(
                        states.get(dep) in {"failed", "blocked"}
                        for dep in self._nodes[node_id].dependencies
                    )
                ]
                for node_id in blocked:
                    states[node_id] = "blocked"
                    metrics[node_id] = GraphNodeMetric(
                        node_id, "blocked", 0, 0,
                        "one or more dependencies did not complete",
                    )
                    pending.remove(node_id)

                ready = [
                    node_id for node_id in order
                    if node_id in pending and all(
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
                    values = tuple(self._resolve(item, outputs) for item in node.inputs)
                    futures[executor.submit(run_node, node, values)] = (node_id, values)
                    pending.remove(node_id)

                for future in as_completed(futures):
                    node_id, values = futures[future]
                    value, elapsed_ns, error = future.result()
                    if error is None and value is not None:
                        outputs[node_id] = value
                        states[node_id] = "completed"
                        metrics[node_id] = GraphNodeMetric(
                            node_id, "completed", elapsed_ns,
                            self._estimate_memory(values, cast(GraphValue, value)),
                        )
                    else:
                        states[node_id] = "failed"
                        metrics[node_id] = GraphNodeMetric(
                            node_id, "failed", elapsed_ns,
                            self._estimate_memory(values), error,
                        )

        return GraphExecutionResult(
            outputs=outputs,
            node_metrics=tuple(metrics[node_id] for node_id in order),
            elapsed_ns=perf_counter_ns() - started,
        )


@dataclass(frozen=True)
class TensorGraphNode:
    """A read-only view of one node in an existing Tensor autograd graph."""

    node_id: str
    operation: str
    shape: tuple[int, ...]
    dependencies: tuple[str, ...]
    requires_grad: bool
    has_backward: bool


@dataclass(frozen=True)
class TensorGraphExecution:
    """Result of executing a supported Tensor trace through ComputeGraph."""

    trace: "TensorGraphTrace" = field(repr=False, compare=False)
    graph_result: GraphExecutionResult
    output_id: str

    @property
    def output_values(self) -> object:
        """Return the graph-computed value for the trace's output tensor."""
        return self.graph_result.output(self.output_id)

    def backward(self) -> None:
        """Backpropagate through the original Tensor graph after graph execution."""
        self.trace.backward()


@dataclass(frozen=True)
class TensorGraphTrace:
    """Topological trace and reverse-mode executor for an eager Tensor graph.

    Forward tensor values are still computed eagerly by Tensor operations.
    This trace provides the shared dependency order used for reverse traversal;
    it does not dispatch tensor kernels through ComputeGraph.
    """

    nodes: tuple[TensorGraphNode, ...]
    output_id: str
    _tensors: tuple[object, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_tensor(cls, output: object) -> "TensorGraphTrace":
        """Capture parents, operation labels, shapes and gradient capabilities."""
        from aria.brain.tensor import Tensor

        if not isinstance(output, Tensor):
            raise TypeError("graph trace output must be a Tensor")
        ordered: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            ordered.append(node)

        visit(output)
        identifiers = {id(node): f"tensor_{index}" for index, node in enumerate(ordered)}
        nodes = tuple(
            TensorGraphNode(
                node_id=identifiers[id(node)],
                operation=node._operation,
                shape=node.shape,
                dependencies=tuple(identifiers[id(parent)] for parent in node._parents),
                requires_grad=node.requires_grad,
                has_backward=node._backward is not None,
            )
            for node in ordered
        )
        return cls(
            nodes=nodes,
            output_id=identifiers[id(output)],
            _tensors=tuple(ordered),
        )

    @property
    def operation_counts(self) -> Mapping[str, int]:
        """Count operations represented by the trace."""
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.operation] = counts.get(node.operation, 0) + 1
        return counts

    def validate(self) -> bool:
        """Check that all dependencies precede their consumers and IDs are unique."""
        identifiers = [node.node_id for node in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            return False
        seen: set[str] = set()
        for node in self.nodes:
            if any(dependency not in seen for dependency in node.dependencies):
                return False
            seen.add(node.node_id)
        return self.output_id in seen

    def execute(
        self,
        runtime: ComputeRuntime | None = None,
        *,
        max_workers: int = 1,
    ) -> "TensorGraphExecution":
        """Execute supported Tensor operations through the general ComputeGraph.

        Leaf tensors become graph constants. The returned execution exposes the
        graph-computed output; backward() delegates to the original Tensor trace.
        Operations with custom or shape-only semantics fail explicitly rather
        than silently falling back to eager execution.
        """
        if not self._tensors or len(self._tensors) != len(self.nodes):
            raise RuntimeError("this trace does not contain an executable tensor graph")
        if not self.validate():
            raise ValueError("cannot execute an invalid tensor graph")

        from aria.brain.tensor import Tensor

        tensors = cast(tuple[Tensor, ...], self._tensors)
        graph = ComputeGraph("tensor-autograd-graph")
        graph_values: dict[str, GraphInput] = {}
        for node, tensor in zip(self.nodes, tensors):
            if node.operation == "leaf":
                graph_values[node.node_id] = _freeze(tensor.to_list())
                continue

            if node.operation not in {"matmul", "add", "multiply", "relu", "sum", "reshape", "bias_add"}:
                raise NotImplementedError(
                    f"Tensor graph execution does not yet support operation "
                    f"{node.operation!r} at {node.node_id!r}"
                )
            inputs = [graph_values[dependency] for dependency in node.dependencies]
            if node.operation == "matmul":
                reference = graph.matmul(node.node_id, inputs[0], inputs[1])
            elif node.operation == "add":
                reference = graph.add(node.node_id, inputs[0], inputs[1])
            elif node.operation == "multiply":
                reference = graph.multiply(node.node_id, inputs[0], inputs[1])
            elif node.operation == "relu":
                reference = graph.relu(node.node_id, inputs[0])
            elif node.operation == "sum":
                reference = graph.sum(node.node_id, inputs[0])
            elif node.operation == "reshape":
                reference = graph.reshape(node.node_id, inputs[0], node.shape)
            else:
                reference = graph.bias_add(node.node_id, inputs[0], inputs[1])
            graph_values[node.node_id] = reference

        result = graph.execute(runtime, max_workers=max_workers)
        return TensorGraphExecution(self, result, self.output_id)

    def backward(self) -> None:
        """Propagate scalar-output gradients in reverse topological order.

        Intermediate gradients are reset for each traversal; leaf gradients
        continue accumulating until their owners call zero_grad().
        """
        from aria.brain.tensor import Tensor

        if not self._tensors or len(self._tensors) != len(self.nodes):
            raise RuntimeError("this trace does not contain an executable tensor graph")
        if not self.validate():
            raise ValueError("cannot execute an invalid tensor graph")
        if any(not isinstance(node, Tensor) for node in self._tensors):
            raise TypeError("tensor graph contains an invalid node")
        tensors = cast(tuple[Tensor, ...], self._tensors)
        output = tensors[-1]
        if output.shape != ():
            raise ValueError("backward() requires a scalar tensor")
        if not output.requires_grad:
            raise ValueError("cannot backpropagate from a tensor without gradients")

        for node in tensors:
            if node._parents:
                node.grad = None
        output.grad = Tensor.scalar(1.0)

        for node in reversed(tensors):
            if node._backward is not None:
                node._backward()
