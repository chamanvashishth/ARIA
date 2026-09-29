from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemStatus:
    name: str
    version: str
    phase: str
    implemented: bool


@dataclass(frozen=True, slots=True)
class Capability:
    name: str
    status: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"IMPLEMENTED", "EXPERIMENTAL", "NOT IMPLEMENTED"}:
            raise ValueError(f"Unsupported capability status: {self.status}")
