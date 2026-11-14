from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


def load_metrics(path: str | Path, cfg: Config) -> pd.DataFrame:
    """Load a metrics CSV (timestamp, cpu_cores, mem_gib, optional rps) onto a regular grid."""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.asfreq(f"{cfg.data.step_seconds}s")
    # Fill short scrape gaps only; long outages stay NaN so they are not learned as real behavior.
    return df.interpolate(limit=6).ffill(limit=12)


def split_point(df: pd.DataFrame, test_fraction: float) -> pd.Timestamp:
    return df.index[int(len(df) * (1 - test_fraction))]
