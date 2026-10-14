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
- `ScheduledComputeBackend` connects this queue to neural forward and backward matrix operations. `matmul_batch()` submits several products to one shared queue, applies dependency ordering and a whole-batch memory budget, and executes synchronously; ordinary eager Tensor calls use the same path as one-item batches.
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
