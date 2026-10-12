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
)
model = TransformerLanguageModel(
    vocab_size=260,
    hidden_size=16,
    intermediate_size=32,
    backend=scheduled,
)
logits = model.forward([10, 11, 12])
print(scheduled.stats)
print(scheduled.commands[-1])
~~~

Each matrix product is submitted and completed synchronously before the caller continues, because the next tensor operation needs the result. This preserves the existing Tensor computation graph: VCPU/VGPU backward matrix products also pass through the adapter. VNPU still rejects gradient-tracked tensors and is usable only after freezing all model parameters with `model.requires_grad_(False)`.

The adapter creates a small queue per operation and retains a bounded history of immutable command snapshots. `clear_history()` resets the history and aggregate counters. Its memory estimate is scheduler accounting metadata, not a measurement of actual Python process memory. Workgroups describe logical output tiles; they are not independently dispatched.

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
- `ScheduledComputeBackend` connects this queue to neural forward and backward matrix operations, executing one complete matrix command synchronously per backend call.
- No asynchronous dispatch, parallel worker pool, or physical hardware scheduling.
- No automatic device discovery or hardware fallback.
- No performance advantage is claimed; VGPU tiling is an algorithmic simulation.
- VNPU quantization is per-matrix symmetric scaling; it is not a calibrated deployment quantizer.
- Optimizer/trainer integration is intended for VCPU/VGPU. VNPU should only be used for frozen forward passes.
- Attention softmax and custom backward calculations remain Python loops; only its score and probability/value matrix products use the selected backend.

## Verification

Tests compare VGPU forward values and gradients against VCPU, exercise Linear and model-level backend injection, verify the VNPU inference-only guard, check frozen TinyLanguageModel and TransformerLanguageModel instances, validate successful/failed runtime profiling, and cover queue dependencies, failure propagation, workgroup coverage, queue limits and memory accounting. Scheduler integration tests compare full Transformer outputs with the direct path, check VCPU/VGPU backpropagation, validate VNPU freezing, enforce memory budgets and bound command history.
