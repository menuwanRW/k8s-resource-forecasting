from __future__ import annotations

import numpy as np
import pandas as pd


def generate(days: int = 28, step_seconds: int = 300, seed: int = 0,
             start: str = "2026-01-05") -> pd.DataFrame:
    """Synthetic workload: daily and weekly seasonality, growth, traffic spikes and a memory leak.

    Useful for trying the pipeline end to end before pointing it at Prometheus.
    """
    rng = np.random.default_rng(seed)
    n = days * 86400 // step_seconds
    idx = pd.date_range(start, periods=n, freq=f"{step_seconds}s")
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    dow = idx.dayofweek.to_numpy()

    daily = 0.55 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.25 * np.sin(4 * np.pi * (hour - 6) / 24)
    weekly = np.where(dow >= 5, -0.35, 0.0)
    trend = np.linspace(0, 0.25, n)
    rps = 220 * (1 + daily + weekly + trend)

    noise = np.zeros(n)
    shocks = rng.normal(0, 8, n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + shocks[i]
    rps = rps + noise

    for _ in range(max(1, days // 2)):  # promotions and batch bursts
        s = int(rng.integers(0, n - 40))
        w = int(rng.integers(6, 30))
        seg = rps[s:s + w]
        seg += rng.uniform(80, 250) * np.hanning(len(seg))
    rps = np.clip(rps, 5, None)

    cpu = 0.012 * rps + 0.25 + rng.normal(0, 0.05, n)
    deploy_cycle = 3 * 86400 // step_seconds  # memory grows until the next deploy resets it
    leak = (np.arange(n) % deploy_cycle) / deploy_cycle * 1.5
    mem = 3.0 + 0.004 * rps + leak + rng.normal(0, 0.05, n)

    return pd.DataFrame(
        {"cpu_cores": np.clip(cpu, 0.05, None), "mem_gib": np.clip(mem, 0.5, None), "rps": rps},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
