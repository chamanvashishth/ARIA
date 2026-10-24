# ARIA compute backends

## Scope

ARIA compute backends are software execution strategies. They currently run on the host CPU and do not access physical accelerator hardware.

| Device | Forward kernel | Autodiff | Intended use |
|---|---|---|---|
| VCPU | Python floating-point matrix multiplication | Yes | Reference execution and default training |
| VGPU | Tiled Python floating-point matrix multiplication | Yes | Experimental tiled execution; correctness comparison |
| VNPU | Symmetric INT8 quantized inputs with integer accumulation | No | Explicitly frozen inference only |

## Neural-engine integration

Tensor.matmul(other, backend=...) accepts two rank-2 tensors and creates a reverse-mode graph node. For VCPU and VGPU, the backward pass computes dA = dC times B-transpose and dB = A-transpose times dC.

Linear uses this primitive for its projection and retains bias-gradient behavior. Leading dimensions are flattened for matrix multiplication and restored on output. The injected backend is propagated through TinyLanguageModel's vocabulary projection and all Transformer Linear projections (Q/K/V, feed-forward and LM head).

Transformer attention score and probability/value matrix multiplications also use the selected backend. Attention softmax and its custom backward logic, embedding lookups, normalization and activation operations remain Python implementations. Backend selection does not mean every model operation runs on the selected device.

## Scheduler integration

`ScheduledComputeBackend` implements the same matrix-backend interface as the direct software backends. Inject it into a Transformer or Linear layer to route forward projections and attention matrix products through the FIFO scheduler:

~~~python
from aria.brain import TransformerLanguageModel
from aria.compute import ComputeRuntime, ScheduledComputeBackend

scheduled = ScheduledComputeBackend(
    ComputeRuntime("vcpu"),
    workgroup_size=8,
    history_limit=512,
    max_workers=4,
)
model = TransformerLanguageModel(
    vocab_size=260,
    hidden_size=16,
    intermediate_size=32,
    backend=scheduled,
)
with scheduled.execution_context("inference") as execution:
    logits = model.forward([10, 11, 12])
print(execution.stats)
print(scheduled.commands[-1])
~~~

The execution context scopes counters, elapsed time, command IDs and peak estimated matrix-buffer reservations to one model call. Contexts are thread-local, cannot be nested on the same thread, and are always cleaned up on exit, including exceptions. The adapter's global command history and aggregate counters are protected for concurrent callers. History remains bounded and can be reset with `clear_history()`.

Cancellation is cooperative: calling `execution.cancel()` causes the next matrix operation to fail before it is submitted. A Python matrix kernel already running cannot be interrupted. The scope reports submitted/completed/failed/cancelled counts and elapsed nanoseconds. Memory figures are scheduler estimates, not process-memory measurements. Workgroups describe logical output tiles; they are not independently dispatched.

`ScheduledComputeBackend.matmul_batch()` submits several matrix products into one shared FIFO queue. Set `max_workers` above one to enable a bounded thread pool for independent commands; the default `max_workers=1` preserves sequential execution. Optional zero-based dependency indices must refer to earlier batch items; failed prerequisites block dependent commands while independent commands continue. All operands are validated and memory reservations are accumulated before execution, so the configured memory budget applies to the whole batch. Results preserve submission order; if any command fails or is blocked, the batch raises after all eligible commands have been processed. `matmul()` delegates to a one-item batch. This public API supports independent matrix workloads, but eager Tensor/Transformer operations still request each result immediately and are not automatically fused into model batches. Each matrix product is completed synchronously before the caller continues, because the next tensor operation needs the result. This preserves the existing Tensor computation graph: VCPU/VGPU backward matrix products also pass through the adapter. VNPU remains inference-only and requires all model parameters to be frozen with `model.requires_grad_(False)`.

Run the end-to-end model diagnostic:

~~~bash
python scripts/benchmark_scheduled_transformer.py --warmup 1 --iterations 5 --sequence-length 8
~~~

It compares direct and scheduler-integrated VCPU Transformer forward passes on identical deterministic model weights and reports numerical equivalence, median latency, command counts and estimated memory. It is an untrained, small synthetic workload and makes no performance claim.

## Training and inference policy

- VCPU is the default and supports gradient tracking.
- VGPU supports gradient tracking through the mathematical matrix-multiplication derivative. It remains a host-CPU emulator.
- VNPU is inference-only because its quantized forward path does not define a straight-through or other gradient estimator. If either matmul operand requires gradients, it raises RuntimeError.
- Freeze every model parameter before VNPU inference:

~~~python
from aria.brain import TinyLanguageModel
from aria.compute import VirtualNPUBackend

model = TinyLanguageModel(
    vocab_size=256,
    embedding_dim=32,
    backend=VirtualNPUBackend(),
)
model.requires_grad_(False)
logits = model.forward([10, 11, 12])
~~~

Calling requires_grad_(True) again on a model using VNPU makes its next projection fail safely instead of silently producing missing gradients.

## Limitations

- No CUDA, Metal, DirectML, OpenCL, or physical GPU/NPU execution.
- Runtime profiling is opt-in and records each matrix operation's backend, operand shapes, duration, and outcome; profiles are held in memory until cleared.
- `python scripts/benchmark_compute.py` compares latency and numerical error against the VCPU reference on deterministic synthetic matrices.
- `CommandQueue` provides FIFO submissions, dependencies on earlier commands, bounded pending capacity, estimated-memory reservations, and logical output workgroup plans.
- `ComputeScheduler` runs ready commands synchronously in submission order, records completed/failed/blocked states, and blocks dependents after prerequisite failures.
- `ScheduledComputeBackend` connects this queue to neural forward and backward matrix operations. `matmul_batch()` submits several products into one shared queue, applies dependency ordering and a whole-batch memory budget, and executes synchronously; ordinary eager Tensor calls use the same path as one-item batches.
- `python scripts/benchmark_compute_batch.py` compares individual calls, a one-worker shared queue, and a four-worker shared queue on identical deterministic VCPU workloads. It reports numerical equivalence and diagnostic latency; Python threads do not guarantee a speedup for CPU-bound kernels.
- `ScheduledExecutionContext` adds per-model-call lifecycle tracking, thread-local scope ownership, cooperative between-kernel cancellation, elapsed-time and resource summaries, and exception-safe cleanup.
- The optional worker pool uses Python threads for independent commands. It is synchronous from the caller's perspective and does not guarantee speedups for CPU-bound Python kernels. Cancellation only applies before a command starts; running kernels cannot be preempted.
- No automatic device discovery or hardware fallback.
- No performance advantage is claimed; VGPU tiling is an algorithmic simulation.
- VNPU quantization is per-matrix symmetric scaling; it is not a calibrated deployment quantizer.
- Optimizer/trainer integration is intended for VCPU/VGPU. VNPU should only be used for frozen forward passes.
- Attention softmax and custom backward calculations remain Python loops; only its score and probability/value matrix products use the selected backend.

## Verification

Tests compare VGPU forward values and gradients against VCPU, exercise Linear and model-level backend injection, verify the VNPU inference-only guard, check frozen TinyLanguageModel and TransformerLanguageModel instances, validate successful/failed runtime profiling, and cover queue dependencies, failure propagation, workgroup coverage, queue limits and memory accounting. Scheduler integration tests compare full Transformer outputs with the direct path, check VCPU/VGPU backpropagation, validate VNPU freezing, enforce memory budgets, bound command history, test scoped lifecycle cleanup, cooperative cancellation and concurrent thread-local contexts.

## Compute graph engine (Block 26)

ComputeGraph represents matrix multiplication as explicit nodes connected by NodeRef values. References may point to nodes added later; validate() checks unknown references and detects cycles before execution. The engine currently supports matrix multiplication nodes only.

execute(runtime, max_workers=...) schedules independent ready nodes in bounded thread-pool waves. A downstream node starts only after its inputs have completed. A failed node blocks its dependents while independent branches continue. The returned GraphExecutionResult contains successful immutable outputs, per-node completion/failure/blocking state, elapsed time and estimated matrix-buffer bytes.

~~~python
from aria.compute import ComputeGraph

graph = ComputeGraph("two-stage-projection")
hidden = graph.matmul("hidden", [[1, 2]], [[3], [4]])
logits = graph.matmul("logits", hidden, [[2, 1]])

result = graph.execute(max_workers=2)
print(result.output(logits))
for metric in result.node_metrics:
    print(metric.node_id, metric.state, metric.elapsed_ns)
~~~

This graph evaluator executes the supported operations in topological dependency order and can run independent ready nodes in bounded worker threads. A failed node blocks its dependents while independent branches continue. It is still an explicit graph API: it does not automatically capture or rerun existing eager Tensor/Transformer calls, and it does not provide graph fusion. All operations execute on the host CPU. Thread scheduling does not guarantee speedups for CPU-bound Python loops, and memory values are estimates rather than process-memory measurements.

## Tensor/autograd graph tracing and reverse-mode execution (Blocks 27–28)

Every Tensor retains an operation label (leaf, matmul, add, multiply, sum, reshape, relu, or custom) alongside the parent links used by reverse-mode autodiff. Call tensor.graph_trace() to get an immutable, inspectable TensorGraphTrace in topological order, including node shapes, dependency IDs, gradient requirements and whether a backward callback exists. The trace can be validated and queried for operation counts.

~~~python
from aria.brain import Parameter, ReLU, VirtualCPUBackend

left = Parameter([[1.0, 2.0]])
right = Parameter([[3.0], [4.0]])
prediction = ReLU().forward(left.matmul(right, backend=VirtualCPUBackend()))
loss = prediction.sum()

trace = loss.graph_trace()
assert trace.validate()
print(trace.operation_counts)
loss.backward()
print(left.gradient_values())
~~~

## Graph-driven reverse-mode autodiff (Block 28)

Tensor.backward() now delegates to TensorGraphTrace.backward(). The trace retains the in-memory tensor nodes, validates their topological dependencies, resets only intermediate gradients, seeds the scalar output gradient, and invokes backward callbacks in reverse topological order. Shared parameters receive accumulated gradients from every dependent branch; leaf gradients continue accumulating across backward() calls until zero_grad().

Forward Tensor operations remain eager: Tensor operations calculate their values immediately, and the trace is built from those recorded dependencies. The general-purpose ComputeGraph evaluator in Block 29 is currently a separate explicit API; it does not yet replace Tensor's eager forward operations or automatically execute Linear/Transformer layers. No graph fusion or hardware acceleration is implied.

~~~python
from aria.brain import Parameter, Tensor

weight = Parameter([[2.0, 3.0]])
scale = Tensor([[2.0, 2.0]])
loss = ((weight * weight) + (weight * scale)).sum()
loss.graph_trace().backward()
print(weight.gradient_values())  # [6.0, 8.0]
~~~


## Tensor graph execution bridge (Block 30)

A Tensor operation trace can now be submitted to the general-purpose ComputeGraph evaluator with `tensor.graph_trace().execute()`. Leaf tensors become graph constants; supported `matmul`, `add`, `multiply`, `relu` and `sum` operations execute through the explicit graph runtime. The returned `TensorGraphExecution.output_values` contains the graph-computed output, and `TensorGraphExecution.backward()` runs reverse-mode autodiff through the original Tensor trace.

~~~python
from aria.brain import Parameter, Tensor
from aria.brain.layers import ReLU

weight = Parameter([[1.0, -2.0], [3.0, 4.0]])
input_values = Tensor([[2.0, 5.0], [-1.0, 6.0]])
loss = ReLU().forward(weight + input_values).sum()

execution = loss.graph_trace().execute(max_workers=2)
print(execution.output_values)
execution.backward()
print(weight.gradient_values())
~~~

This bridge deliberately rejects reshape and custom Tensor operations with a clear `NotImplementedError`; it never silently switches back to eager execution. Consequently, full Linear/Transformer graph execution is not yet supported because those paths currently include reshape and custom bias/attention operations. Forward graph execution and reverse-mode autodiff are connected through a result wrapper, but the graph evaluator does not yet replace the full neural-model execution path. All kernels still run on the host CPU; worker threads do not guarantee speedups.


## Linear graph integration (Block 31)

The explicit Tensor graph executor now supports the `reshape` and `bias_add` operations used by `Linear`. The layer labels its bias operation explicitly, allowing a complete Linear forward path—including input flattening, projection, bias, and output-shape restoration—to run through ComputeGraph.

~~~python
from aria.brain import Linear, Parameter

layer = Linear(2, 2, seed=9)
inputs = Parameter([[[1.0, 2.0], [3.0, 4.0]]])
loss = layer.forward(inputs).sum()

execution = loss.graph_trace().execute()
print(execution.output_values)
execution.backward()
print(layer.weight.gradient_values())
print(layer.bias.gradient_values())
~~~

The graph executor computes forward values for these operations; the execution result's `backward()` method delegates gradient propagation to the original Tensor autograd graph. Tests compare graph and eager scalar outputs and verify input, weight, and bias gradients. Custom attention, embedding lookup and normalization operations remain outside this graph executor. All compute remains host-CPU software execution.


## Multi-layer Sequential graph execution (Block 32)

Module.graph_execute(inputs) runs a module (including Sequential) once, captures its Tensor trace, and executes supported operations through ComputeGraph. The result retains the model output shape for inference. For training, pass a scalar objective callback because the current reverse-mode interface requires a scalar loss.

~~~python
from aria.brain import Linear, ReLU, SGD, Sequential, Tensor

model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
inputs = Tensor([[1.0, 2.0], [2.0, -1.0]])

# Graph-executed inference returns the full (2, 1) output.
inference = model.graph_execute(inputs)
print(inference.output_values)

# Graph-executed training objective and existing autodiff/optimizer.
training = model.graph_execute(inputs, objective=lambda output: output.sum())
training.backward()
optimizer = SGD(model.parameters(), learning_rate=0.01)
optimizer.step()
~~~

The graph executor computes the supported forward operations; TensorGraphExecution.backward() delegates gradients to the original Tensor reverse-mode graph. Integration tests compare multi-layer graph output against eager output, verify gradients reach both Linear layers and the input, and verify SGD updates every trainable parameter. All execution remains host-CPU software; this is not physical accelerator execution or graph fusion.


## Transformer graph integration status (Block 33)

The Transformer feed-forward MLP (Linear → ReLU → Linear) now has integration coverage through Module.graph_execute. Its graph output is checked against eager execution, and a graph-executed scalar objective is backpropagated through the existing Tensor autodiff path before SGD updates every feed-forward parameter.

This is a deliberate component-level milestone, not a claim that the full Transformer runs through ComputeGraph yet. RMSNorm, token Embedding, causal self-attention, and cross-entropy still use custom Tensor operations without graph-native forward kernels. Full Transformer graph execution remains unsupported until those operations receive explicit graph implementations. All existing Transformer forward/training behavior remains available through the eager Tensor path.

## Graph-native Transformer operations

The explicit compute graph also supports Embedding gathers, RMSNorm, single-head causal attention, and stable mean cross-entropy. These operations can be reached from an eager Tensor computation through `Tensor.graph_trace().execute()`. The graph computes the forward values; `TensorGraphExecution.backward()` delegates gradient propagation to the original reverse-mode Tensor graph. This preserves the tested eager gradient rules while graph execution is introduced incrementally.

Causal attention receives explicit batch/time metadata so the graph keeps each sample's causal history isolated. The graph operations use host-CPU Python implementations; they do not imply physical GPU/NPU execution. The current graph trace is not yet an independent graph compiler or graph-native autodiff engine.


## Complete Transformer graph execution (Block 34)

TransformerLanguageModel.forward(...).graph_trace().execute() executes supported single-sequence inference through ComputeGraph. Batched inference supports batch sizes including one; causal-attention graph metadata explicitly preserves whether the original input carried a batch dimension.

For training, use TransformerLanguageModel.graph_execute_loss(token_batches, target_batches). This constructs the normal scalar cross-entropy loss and executes the complete supported forward trace in ComputeGraph. Call execution.backward() to propagate gradients through the existing Tensor autodiff callbacks, then use the standard optimizer.

~~~python
from aria.brain import SGD, TransformerLanguageModel

model = TransformerLanguageModel(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_layers=2,
    max_sequence_length=32,
    seed=7,
)
inputs = [[1, 2, 3], [4, 5, 6]]
targets = [[2, 3, 4], [5, 6, 7]]

execution = model.graph_execute_loss(inputs, targets, max_workers=2)
print("loss:", execution.output_values)
execution.backward()
optimizer = SGD(model.parameters(), learning_rate=0.001)
optimizer.step()
~~~

Integration tests compare graph and eager logits for unbatched input and batches of size one and two, then compare graph and eager loss, verify all model parameters receive gradients, and verify an optimizer update. Graph forward kernels and attention remain host-CPU software implementations. Backward is still delegated to the original Tensor reverse-mode callbacks; this is not a standalone graph compiler, fused execution, or physical GPU/NPU acceleration.
