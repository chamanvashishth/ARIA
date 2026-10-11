"""Software-emulated compute backends for ARIA's local numerical kernels.

All current backends execute on the host CPU. The virtual GPU and NPU classes
model different kernel strategies and numeric formats; they do not provide
physical accelerator access or hardware acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


DeviceKind = Literal["vcpu", "vgpu", "vnpu"]
Matrix = Sequence[Sequence[int | float]]
_MatrixValues = list[list[float]]


@dataclass(frozen=True)
class BackendInfo:
    """Honest capability description for one software compute backend."""

    device: DeviceKind
    name: str
    execution: str
    numeric_mode: str
    hardware_acceleration: bool = False
    supports_forward_matmul: bool = True
    supports_autodiff: bool = False


class ComputeBackend(Protocol):
    """Interface shared by ARIA's software compute backends."""

    @property
    def info(self) -> BackendInfo:
        ...

    def matmul(self, left: Matrix, right: Matrix) -> _MatrixValues:
        ...


def _normalize_matrix(values: Matrix, name: str) -> _MatrixValues:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a non-empty matrix")
    if not values:
        raise ValueError(f"{name} must not be empty")

    normalized: _MatrixValues = []
    width: int | None = None
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} row {row_index} must be a sequence")
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError(f"{name} rows must not be empty")
        elif len(row) != width:
            raise ValueError(f"{name} must be rectangular")

        normalized_row: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} values must be real numbers")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} values must be finite")
            normalized_row.append(numeric)
        normalized.append(normalized_row)
    return normalized


def _validated_operands(left: Matrix, right: Matrix) -> tuple[_MatrixValues, _MatrixValues]:
    a = _normalize_matrix(left, "left operand")
    b = _normalize_matrix(right, "right operand")
    if len(a[0]) != len(b):
        raise ValueError(
            "matrix dimensions do not match: "
            f"{len(a)}x{len(a[0])} cannot multiply {len(b)}x{len(b[0])}"
        )
    return a, b


class VirtualCPUBackend:
    """Pure-Python floating-point reference backend executed on the host CPU."""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            device="vcpu",
            name="ARIA Virtual CPU",
            execution="pure-python-reference",
            numeric_mode="python-float",
        )

    def matmul(self, left: Matrix, right: Matrix) -> _MatrixValues:
        a, b = _validated_operands(left, right)
        rows, inner, columns = len(a), len(b), len(b[0])
        return [
            [
                sum(a[row][index] * b[index][column] for index in range(inner))
                for column in range(columns)
            ]
            for row in range(rows)
        ]


class VirtualGPUBackend:
    """Tiled matrix-multiplication simulator executed by host-CPU loops."""

    def __init__(self, tile_size: int = 16) -> None:
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        self.tile_size = tile_size

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            device="vgpu",
            name="ARIA Virtual GPU Emulator",
            execution=f"host-cpu-tiled-emulation:{self.tile_size}",
            numeric_mode="python-float",
        )

    def matmul(self, left: Matrix, right: Matrix) -> _MatrixValues:
        a, b = _validated_operands(left, right)
        rows, inner, columns = len(a), len(b), len(b[0])
        output = [[0.0] * columns for _ in range(rows)]
        tile = self.tile_size

        for row_start in range(0, rows, tile):
            for column_start in range(0, columns, tile):
                for inner_start in range(0, inner, tile):
                    for row in range(row_start, min(row_start + tile, rows)):
                        for column in range(column_start, min(column_start + tile, columns)):
                            output[row][column] += sum(
                                a[row][index] * b[index][column]
                                for index in range(inner_start, min(inner_start + tile, inner))
                            )
        return output


class VirtualNPUBackend:
    """Symmetric int8 matrix-multiplication simulator using CPU integer loops."""

    _MAX_INT32 = 2**31 - 1
    _QUANT_MAX = 127

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            device="vnpu",
            name="ARIA Virtual NPU Emulator",
            execution="host-cpu-int8-emulation",
            numeric_mode="symmetric-int8-quantized-inputs/int32-accumulation",
        )

    @classmethod
    def _quantize(cls, values: _MatrixValues) -> tuple[list[list[int]], float]:
        maximum = max(abs(value) for row in values for value in row)
        scale = maximum / cls._QUANT_MAX if maximum else 1.0
        quantized = [
            [
                max(-cls._QUANT_MAX, min(cls._QUANT_MAX, round(value / scale)))
                for value in row
            ]
            for row in values
        ]
        return quantized, scale

    def matmul(self, left: Matrix, right: Matrix) -> _MatrixValues:
        a, b = _validated_operands(left, right)
        rows, inner, columns = len(a), len(b), len(b[0])
        if inner * self._QUANT_MAX * self._QUANT_MAX > self._MAX_INT32:
            raise ValueError("inner dimension exceeds the virtual NPU int32 accumulator limit")

        quantized_a, scale_a = self._quantize(a)
        quantized_b, scale_b = self._quantize(b)
        scale = scale_a * scale_b
        return [
            [
                float(sum(
                    quantized_a[row][index] * quantized_b[index][column]
                    for index in range(inner)
                )) * scale
                for column in range(columns)
            ]
            for row in range(rows)
        ]


class ComputeRuntime:
    """Select and expose one explicit software compute backend."""

    def __init__(self, device: DeviceKind = "vcpu", *, tile_size: int = 16) -> None:
        if device == "vcpu":
            self.backend: ComputeBackend = VirtualCPUBackend()
        elif device == "vgpu":
            self.backend = VirtualGPUBackend(tile_size=tile_size)
        elif device == "vnpu":
            self.backend = VirtualNPUBackend()
        else:
            raise ValueError(f"unsupported compute device: {device}")

    @property
    def info(self) -> BackendInfo:
        return self.backend.info

    @staticmethod
    def available_devices() -> tuple[BackendInfo, ...]:
        return (
            VirtualCPUBackend().info,
            VirtualGPUBackend().info,
            VirtualNPUBackend().info,
        )

    def matmul(self, left: Matrix, right: Matrix) -> _MatrixValues:
        """Multiply two matrices using the selected software backend."""
        return self.backend.matmul(left, right)
