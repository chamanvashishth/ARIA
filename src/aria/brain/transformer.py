"""Small decoder-only Transformer language model for ARIA."""

from __future__ import annotations

import math
import random

from aria.brain.layers import Embedding, Linear
from aria.brain.module import Module
from aria.brain.tensor import Tensor


def _reshape(values: list[float], shape: tuple[int, ...]) -> object:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    width = math.prod(shape[1:])
    return [_reshape(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("matrix dimensions do not match")
    return [[sum(row[k] * b[k][j] for k in range(len(b[0]))) for j in range(len(b[0]))] for row in a]


class RMSNorm(Module):
    """RMS normalization with trainable scale."""

    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        from aria.brain.parameter import Parameter
        self.weight = Parameter([1.0] * dimension)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) < 2:
            raise ValueError("RMSNorm expects [..., hidden]")
        hidden = x.shape[-1]
        rows = math.prod(x.shape[:-1])
        xv = x._values
        out = []
        inv_rms = []
        for row in range(rows):
            start = row * hidden
            rms = math.sqrt(sum(v * v for v in xv[start:start + hidden]) / hidden + self.eps)
            inv_rms.append(1.0 / rms)
            out.extend(xv[start + col] * inv_rms[-1] * self.weight._values[col] for col in range(hidden))

        def backward(result: Tensor) -> None:
            grad = result.grad._values
            dx = [0.0] * len(xv)
            dw = [0.0] * hidden
            for row in range(rows):
                start = row * hidden
                dot = sum(grad[start + c] * self.weight._values[c] * xv[start + c] for c in range(hidden))
                inv = inv_rms[row]
                inv3 = inv ** 3
                for c in range(hidden):
                    dw[c] += grad[start + c] * xv[start + c] * inv
                    dx[start + c] += (
                        grad[start + c] * self.weight._values[c] * inv
                        - xv[start + c] * dot * inv3 / hidden
                    )
            if x.requires_grad:
                x._accumulate(dx)
            self.weight._accumulate(dw)

        return Tensor.operation(_reshape(out, x.shape), parents=(x, self.weight), backward=backward)



class CausalSelfAttention(Module):
    """Single-head causal self-attention."""

    def __init__(self, hidden_size: int, *, seed: int | None = None) -> None:
        self.hidden_size = hidden_size
        self.query = Linear(hidden_size, hidden_size, seed=seed)
        self.key = Linear(hidden_size, hidden_size, seed=None if seed is None else seed + 1)
        self.value = Linear(hidden_size, hidden_size, seed=None if seed is None else seed + 2)
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) not in (2, 3):
            raise ValueError("attention expects [time, hidden] or [batch, time, hidden]")
        batched = len(x.shape) == 3
        batch, time, hidden = (x.shape if batched else (1, *x.shape))
        if hidden != self.hidden_size or batch <= 0 or time <= 0:
            raise ValueError("attention input dimensions are invalid")
        flat_x = x.reshape((batch * time, hidden)) if batched else x
        q_tensor = self.query.forward(flat_x)
        k_tensor = self.key.forward(flat_x)
        v_tensor = self.value.forward(flat_x)
        q, k, v = q_tensor._values, k_tensor._values, v_tensor._values
        outputs = [0.0] * (batch * time * hidden)
        cache: list[tuple[int, int, list[float]]] = []

        for b in range(batch):
            offset = b * time * hidden
            for i in range(time):
                scores = [
                    sum(q[offset + i * hidden + d] * k[offset + j * hidden + d] for d in range(hidden)) * self.scale
                    if j <= i else -float("inf")
                    for j in range(time)
                ]
                maximum = max(scores[:i + 1])
                exp_scores = [math.exp(s - maximum) if j <= i else 0.0 for j, s in enumerate(scores)]
                normalizer = sum(exp_scores)
                probs = [value / normalizer for value in exp_scores]
                for d in range(hidden):
                    outputs[offset + i * hidden + d] = sum(
                        probs[j] * v[offset + j * hidden + d] for j in range(i + 1)
                    )
                cache.append((b, i, probs))

        def backward(result: Tensor) -> None:
            grad_out = result.grad._values
            dq = [0.0] * len(q)
            dk = [0.0] * len(k)
            dv = [0.0] * len(v)
            for b, i, probs in cache:
                offset = b * time * hidden
                dprob = [0.0] * time
                for j in range(i + 1):
                    for d in range(hidden):
                        go = grad_out[offset + i * hidden + d]
                        dv[offset + j * hidden + d] += probs[j] * go
                        dprob[j] += go * v[offset + j * hidden + d]
                dot = sum(dprob[j] * probs[j] for j in range(i + 1))
                for j in range(i + 1):
                    ds = (dprob[j] - dot) * probs[j]
                    for d in range(hidden):
                        dq[offset + i * hidden + d] += ds * k[offset + j * hidden + d] * self.scale
                        dk[offset + j * hidden + d] += ds * q[offset + i * hidden + d] * self.scale

            q_tensor._accumulate(dq)
            k_tensor._accumulate(dk)
            v_tensor._accumulate(dv)

        output_shape = (batch, time, hidden) if batched else (time, hidden)
        return Tensor.operation(_reshape(outputs, output_shape), parents=(q_tensor, k_tensor, v_tensor), backward=backward)



class FeedForward(Module):
    """Two-layer MLP used inside a Transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, seed: int | None = None) -> None:
        from aria.brain.layers import ReLU
        self.up = Linear(hidden_size, intermediate_size, seed=seed)
        self.activation = ReLU()
        self.down = Linear(intermediate_size, hidden_size, seed=None if seed is None else seed + 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.down.forward(self.activation.forward(self.up.forward(x)))


class TransformerBlock(Module):
    """Pre-norm decoder block with causal self-attention and MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, seed: int | None = None) -> None:
        self.norm1 = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, seed=seed)
        self.norm2 = RMSNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, seed=seed)

    def forward(self, x: Tensor) -> Tensor:
        attended = self.attention.forward(self.norm1.forward(x))
        residual = x + attended
        return residual + self.feed_forward.forward(self.norm2.forward(residual))


class TransformerLanguageModel(Module):
    """Small decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int = 1,
        max_sequence_length: int = 128,
        *,
        seed: int | None = None,
    ) -> None:
        if min(vocab_size, hidden_size, intermediate_size, num_layers, max_sequence_length) <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.token_embedding = Embedding(vocab_size, hidden_size, seed=seed)
        rng = random.Random(seed)
        self.position_embedding = Embedding(max_sequence_length, hidden_size, seed=rng.randrange(1_000_000))
        self.layers = [
            TransformerBlock(hidden_size, intermediate_size, seed=rng.randrange(1_000_000))
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(hidden_size)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False, seed=rng.randrange(1_000_000))

    def forward(self, token_ids: list[int]) -> Tensor:
        if not token_ids:
            raise ValueError("token sequence cannot be empty")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("sequence exceeds model context length")
        token = self.token_embedding.forward(token_ids)
        position = self.position_embedding.forward(list(range(len(token_ids))))
        hidden = token + position
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return self.lm_head.forward(self.norm.forward(hidden))

    def forward_batch(self, token_batches: list[list[int]]) -> Tensor:
        """Run a fixed-length batch with independent causal attention per item."""
        if not token_batches:
            raise ValueError("batch must contain at least one sequence")
        sequence_length = len(token_batches[0])
        if sequence_length == 0:
            raise ValueError("token sequences cannot be empty")
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence exceeds model context length")
        if any(len(sequence) != sequence_length for sequence in token_batches):
            raise ValueError("all batch sequences must have the same length")
        batch_size = len(token_batches)
        flat_tokens = [token for sequence in token_batches for token in sequence]
        flat_positions = list(range(sequence_length)) * batch_size
        token = self.token_embedding.forward(flat_tokens).reshape((batch_size, sequence_length, self.hidden_size))
        position = self.position_embedding.forward(flat_positions).reshape((batch_size, sequence_length, self.hidden_size))
        hidden = token + position
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return self.lm_head.forward(self.norm.forward(hidden))

    def loss_batch(
        self,
        token_batches: list[list[int]],
        target_batches: list[list[int]],
    ) -> Tensor:
        """Mean next-token cross-entropy over a fixed-length batch."""
        if len(token_batches) != len(target_batches) or not token_batches:
            raise ValueError("input and target batches must have equal non-zero size")
        if any(len(inputs) != len(targets) for inputs, targets in zip(token_batches, target_batches)):
            raise ValueError("each input sequence must match its target length")
        logits = self.forward_batch(token_batches)
        batch_size, sequence_length, vocab_size = logits.shape
        flat_logits = logits.reshape((batch_size * sequence_length, vocab_size))
        flat_targets = [target for sequence in target_batches for target in sequence]
        from aria.brain.language_model import _log_softmax_loss
        return _log_softmax_loss(flat_logits, flat_targets)
