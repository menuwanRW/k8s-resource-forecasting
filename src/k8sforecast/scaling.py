from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import ScalingConfig


@dataclass
class Recommendation:
    pods: int
    nodes: int
    current_pods: int
    current_nodes: int
    pod_action: str
    node_action: str
    cpu_peak_p90: float
    mem_peak_p90_gib: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class CapacityPlanner:
    """Turns a demand forecast into a pod count (the HPA formula) and a node count (bin packing)."""

    def __init__(self, cfg: ScalingConfig):
        self.cfg = cfg

    def pods_needed(self, cpu_cores, mem_gib):
        """Works on scalars and numpy arrays."""
        c = self.cfg
        by_cpu = np.asarray(cpu_cores) / (c.pod_cpu_request * c.target_cpu_util)
        by_mem = np.asarray(mem_gib) / (c.pod_mem_request_gib * c.target_mem_util)
        pods = np.nan_to_num(np.ceil(np.maximum(by_cpu, by_mem)), nan=c.min_pods)  # missing data -> floor
        return np.clip(pods, c.min_pods, c.max_pods).astype(int)

    def nodes_needed(self, pods: int) -> int:
        c = self.cfg
        usable = 1 - c.node_headroom
        by_cpu = pods * c.pod_cpu_request / (c.node_cpu_alloc * usable)
        by_mem = pods * c.pod_mem_request_gib / (c.node_mem_alloc_gib * usable)
        by_slots = pods / c.max_pods_per_node
        nodes = int(np.ceil(max(by_cpu, by_mem, by_slots)))
        return int(np.clip(nodes, c.min_nodes, c.max_nodes))

    def _guard(self, desired: int, current: int, steps_since_change: int) -> tuple[int, str]:
        """Scale up immediately; scale down only after a cooldown and in limited steps."""
        c = self.cfg
        if desired > current:
            return desired, "scale up"
        if desired == current:
            return current, "hold"
        if steps_since_change < c.scale_down_cooldown_steps:
            return current, "hold (scale-down cooldown)"
        step = max(1, int(current * c.max_scale_down_fraction))
        return max(desired, current - step), "scale down"

    def recommend(self, forecast: pd.DataFrame, current_pods: int, current_nodes: int,
                  steps_since_change: int = 10**9) -> Recommendation:
        c = self.cfg
        window = forecast[forecast.index <= c.lead_time_steps]
        if window.empty:
            window = forecast.iloc[:1]
        cpu_peak = float(window["cpu_cores_q0.9"].max())
        mem_peak = float(window["mem_gib_q0.9"].max())

        desired_pods = int(self.pods_needed(cpu_peak, mem_peak))
        pods, pod_action = self._guard(desired_pods, current_pods, steps_since_change)
        desired_nodes = self.nodes_needed(pods)
        nodes, node_action = self._guard(desired_nodes, current_nodes, steps_since_change)

        reasons = [
            f"p90 peak over the next {c.lead_time_steps} steps: {cpu_peak:.2f} cores, {mem_peak:.2f} GiB",
            f"pods wanted {desired_pods}, nodes wanted {desired_nodes}",
        ]
        return Recommendation(pods, nodes, current_pods, current_nodes, pod_action, node_action,
                              cpu_peak, mem_peak, reasons)
