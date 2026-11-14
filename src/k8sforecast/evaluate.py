from __future__ import annotations

import numpy as np
import pandas as pd

from .models import QuantileForecaster
from .scaling import CapacityPlanner


def smape(y: np.ndarray, p: np.ndarray) -> float:
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(np.mean(np.abs(y - p) / np.where(denom == 0, 1, denom)) * 100)


def pinball(y: np.ndarray, p: np.ndarray, q: float) -> float:
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate_models(model: QuantileForecaster, df: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """Accuracy per target and horizon on the held-out period, against two naive baselines."""
    cfg, spd = model.cfg, model.cfg.steps_per_day
    hi = max(model.quantiles)
    rows = []
    for target in cfg.model.targets:
        for h in cfg.model.horizons:
            pred = model.predict_frame(df, target, h)
            y = df[target].shift(-h)
            persist = df[target]                       # "it stays as it is now"
            seasonal = df[target].shift(spd - h)       # "same time yesterday"
            mask = (df.index >= test_start) & y.notna() & seasonal.notna()
            yy = y[mask].to_numpy()
            rows.append({
                "target": target, "horizon_min": h * cfg.data.step_seconds // 60,
                "mae": np.mean(np.abs(yy - pred.loc[mask, "q0.5"])),
                "mae_persistence": np.mean(np.abs(yy - persist[mask])),
                "mae_yesterday": np.mean(np.abs(yy - seasonal[mask])),
                "smape_%": smape(yy, pred.loc[mask, "q0.5"].to_numpy()),
                f"coverage_q{hi}": float(np.mean(yy <= pred.loc[mask, f"q{hi}"])),
                f"pinball_q{hi}": pinball(yy, pred.loc[mask, f"q{hi}"].to_numpy(), hi),
            })
    return pd.DataFrame(rows).round(4)


def backtest_scaling(model: QuantileForecaster, df: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """Replay the test period: pods ordered at time t must be ready at t + lead time.

    Reactive policy sizes from the load it sees at t. Forecast policy sizes from the p90 peak
    over the lead time. Hysteresis is left out so only forecast quality is compared.
    """
    cfg = model.cfg
    lead = cfg.scaling.lead_time_steps
    hs = [h for h in cfg.model.horizons if h <= lead]
    if not hs:
        raise ValueError("lead_time_steps must be >= the smallest forecast horizon")
    planner = CapacityPlanner(cfg.scaling)

    peak = {}
    for target in ("cpu_cores", "mem_gib"):
        frames = [model.predict_frame(df, target, h)["q0.9"] for h in hs]
        peak[target] = pd.concat(frames, axis=1).max(axis=1)

    fc_pods = planner.pods_needed(peak["cpu_cores"].to_numpy(), peak["mem_gib"].to_numpy())
    re_pods = planner.pods_needed(df["cpu_cores"].to_numpy(), df["mem_gib"].to_numpy())
    required = planner.pods_needed(df["cpu_cores"].shift(-lead).to_numpy(),
                                   df["mem_gib"].shift(-lead).to_numpy())
    valid = (df.index >= test_start) & df["cpu_cores"].shift(-lead).notna().to_numpy() \
        & peak["cpu_cores"].notna().to_numpy()

    rows = []
    for name, pods in (("reactive", re_pods), ("forecast_p90", fc_pods)):
        short = (pods < required)[valid]
        gap = (required - pods)[valid]
        rows.append({
            "policy": name,
            "under_provisioned_%": round(float(short.mean() * 100), 2),
            "avg_shortfall_pods": round(float(gap[short].mean()) if short.any() else 0.0, 2),
            "avg_pods": round(float(pods[valid].mean()), 2),
            "avg_surplus_pods": round(float(np.clip(-gap, 0, None).mean()), 2),
        })
    return pd.DataFrame(rows)
