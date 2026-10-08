"""Core neural primitives for ARIA's trainable brain foundation."""

from aria.brain.layers import Embedding, Linear, ReLU, Sequential
from aria.brain.language_model import TinyLanguageModel
from aria.brain.module import Module
from aria.brain.transformer import TransformerLanguageModel
from aria.brain.optim import ExponentialDecay, LearningRateScheduler, SGD, StepDecay
from aria.brain.parameter import Parameter
from aria.brain.tensor import Tensor

__all__ = [
    "Embedding",
    "Linear",
    "Module",
    "Parameter",
    "ReLU",
    "ExponentialDecay",
    "LearningRateScheduler",
    "SGD",
    "StepDecay",
    "Sequential",
    "Tensor",
    "TinyLanguageModel",
    "TransformerLanguageModel",
]
