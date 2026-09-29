from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Machine-readable snapshot used by feasibility and scheduling layers."""

    operating_system: str
    cpu_name: str
    cpu_architecture: str
    cpu_cores: int | None
    cpu_threads: int | None
    system_ram_bytes: int | None
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_bytes: int | None
    gpu_backend: str | None
    npu_available: bool
    npu_name: str | None
    npu_runtime: str | None
    storage_free_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "operating_system": self.operating_system,
            "cpu_name": self.cpu_name,
            "cpu_architecture": self.cpu_architecture,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "system_ram_bytes": self.system_ram_bytes,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "gpu_backend": self.gpu_backend,
            "npu_available": self.npu_available,
            "npu_name": self.npu_name,
            "npu_runtime": self.npu_runtime,
            "storage_free_bytes": self.storage_free_bytes,
        }
