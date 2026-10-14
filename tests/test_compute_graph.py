import pytest

from aria.compute import ComputeGraph, ComputeRuntime, NodeRef


def test_graph_executes_chain_and_returns_immutable_outputs() -> None:
    graph = ComputeGraph("chain")
    first = graph.matmul("first", [[1, 2]], [[3], [4]])
    second = graph.matmul("second", first, [[2]])
    result = graph.execute()

    assert result.output(first) == ((11.0,),)
    assert result.output(second) == ((22.0,),)
    assert [item.state for item in result.node_metrics] == ["completed", "completed"]
    assert all(item.elapsed_ns >= 0 for item in result.node_metrics)
    assert all(item.estimated_memory_bytes > 0 for item in result.node_metrics)


def test_graph_supports_forward_references_and_topological_order() -> None:
    graph = ComputeGraph()
    downstream = graph.matmul("downstream", NodeRef("upstream"), [[2]])
    graph.matmul("upstream", [[3]], [[4]])

    assert graph.validate() == ("upstream", "downstream")
    assert graph.execute().output(downstream) == ((24.0,),)


def test_graph_runs_independent_nodes_concurrently() -> None:
    import time
    from threading import Lock

    class TrackingRuntime:
        def __init__(self):
            self._runtime = ComputeRuntime()
            self.info = self._runtime.info
            self.lock = Lock()
            self.active = 0
            self.maximum_active = 0

        def matmul(self, left, right):
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.03)
            try:
                return self._runtime.matmul(left, right)
            finally:
                with self.lock:
                    self.active -= 1

    runtime = TrackingRuntime()
    graph = ComputeGraph()
    graph.matmul("a", [[1]], [[2]])
    graph.matmul("b", [[2]], [[3]])
    graph.matmul("c", [[3]], [[4]])

    result = graph.execute(runtime, max_workers=3)

    assert result.output("a") == ((2.0,),)
    assert result.output("b") == ((6.0,),)
    assert result.output("c") == ((12.0,),)
    assert runtime.maximum_active > 1


def test_graph_blocks_only_dependents_after_node_failure() -> None:
    class SelectiveRuntime:
        def __init__(self):
            self._runtime = ComputeRuntime()
            self.info = self._runtime.info

        def matmul(self, left, right):
            if left[0][0] == 9:
                raise ArithmeticError("deliberate graph failure")
            return self._runtime.matmul(left, right)

    graph = ComputeGraph()
    failed = graph.matmul("failed", [[9]], [[2]])
    graph.matmul("blocked", failed, [[3]])
    graph.matmul("independent", [[4]], [[5]])

    result = graph.execute(SelectiveRuntime(), max_workers=2)

    states = {item.node_id: item.state for item in result.node_metrics}
    assert states == {"failed": "failed", "blocked": "blocked", "independent": "completed"}
    assert "ArithmeticError" in result.node_metrics[0].error
    assert result.output("independent") == ((20.0,),)
    with pytest.raises(KeyError, match="no successful output"):
        result.output("blocked")


def test_graph_rejects_missing_references_and_cycles() -> None:
    missing = ComputeGraph()
    missing.matmul("orphan", NodeRef("not-added"), [[1]])
    with pytest.raises(ValueError, match="unknown nodes"):
        missing.validate()

    cyclic = ComputeGraph()
    cyclic.matmul("a", NodeRef("b"), [[1]])
    cyclic.matmul("b", NodeRef("a"), [[1]])
    with pytest.raises(ValueError, match="cycle"):
        cyclic.execute()


def test_graph_rejects_duplicate_ids_and_invalid_inputs() -> None:
    graph = ComputeGraph()
    graph.matmul("same", [[1]], [[2]])
    with pytest.raises(ValueError, match="duplicate"):
        graph.matmul("same", [[1]], [[2]])
    with pytest.raises(ValueError, match="non-empty matrix"):
        graph.matmul("empty", [], [[1]])
    with pytest.raises(ValueError, match="graph name"):
        ComputeGraph(" ")


def test_graph_validates_workers_and_matrix_shapes_at_execution() -> None:
    graph = ComputeGraph()
    graph.matmul("bad-shape", [[1, 2]], [[1]])
    with pytest.raises(ValueError, match="max_workers"):
        graph.execute(max_workers=0)

    result = graph.execute()
    assert result.node_metrics[0].state == "failed"
    assert "dimensions do not match" in result.node_metrics[0].error
