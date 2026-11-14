from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    step_seconds: int = 300
    prometheus_url: str = "http://localhost:9090"
    namespace: str = "default"
    workload: str = "java-backend"
    lookback_days: int = 28
    include_rps: bool = True


@dataclass
class ModelConfig:
    targets: list[str] = field(default_factory=lambda: ["cpu_cores", "mem_gib"])
    exog: list[str] = field(default_factory=lambda: ["rps"])
    horizons: list[int] = field(default_factory=lambda: [1, 3, 6, 12])  # in steps
    quantiles: list[float] = field(default_factory=lambda: [0.5, 0.9])
    lags: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 6, 12, 24])
    windows: list[int] = field(default_factory=lambda: [6, 12, 36])
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    test_fraction: float = 0.2


@dataclass
class ScalingConfig:
    pod_cpu_request: float = 0.5
    pod_mem_request_gib: float = 1.0
    target_cpu_util: float = 0.7
    target_mem_util: float = 0.8
    min_pods: int = 2
    max_pods: int = 50
    node_cpu_alloc: float = 7.5
    node_mem_alloc_gib: float = 28.0
    max_pods_per_node: int = 50
    node_headroom: float = 0.15
    min_nodes: int = 2
    max_nodes: int = 30
    lead_time_steps: int = 6
    scale_down_cooldown_steps: int = 12
    max_scale_down_fraction: float = 0.25


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    @property
    def steps_per_day(self) -> int:
        return 86400 // self.data.step_seconds

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            scaling=ScalingConfig(**raw.get("scaling", {})),
        )
