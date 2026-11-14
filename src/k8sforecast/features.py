from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def feature_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    return list(cfg.model.targets) + [c for c in cfg.model.exog if c in df.columns]


def base_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Features known at time t: lags, rolling stats and short-term trend. Nothing looks ahead."""
    feats: dict[str, pd.Series] = {}
    for col in feature_columns(df, cfg):
        s = df[col]
        for k in cfg.model.lags:
            feats[f"{col}_lag{k}"] = s.shift(k)
        for w in cfg.model.windows:
            r = s.rolling(w, min_periods=max(2, w // 2))
            feats[f"{col}_mean{w}"] = r.mean()
            feats[f"{col}_std{w}"] = r.std()
            feats[f"{col}_max{w}"] = r.max()
        feats[f"{col}_diff1"] = s.diff()
        feats[f"{col}_slope12"] = (s - s.shift(12)) / 12
    return pd.DataFrame(feats, index=df.index)


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    hod = index.hour.to_numpy() + index.minute.to_numpy() / 60
    dow = index.dayofweek.to_numpy()
    return pd.DataFrame({
        "hod_sin": np.sin(2 * np.pi * hod / 24),
        "hod_cos": np.cos(2 * np.pi * hod / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(float),
    })


def horizon_features(df: pd.DataFrame, target: str, h: int, cfg: Config) -> pd.DataFrame:
    """Features aligned to the time being predicted (t + h): its calendar position and the
    value at the same time yesterday and last week. Those past values are already known at t."""
    spd = cfg.steps_per_day
    if h >= spd:
        raise ValueError("horizon must be shorter than one day")
    target_time = df.index + pd.Timedelta(seconds=h * cfg.data.step_seconds)
    cal = _calendar(pd.DatetimeIndex(target_time))
    cal.index = df.index
    s = df[target]
    cal[f"{target}_same_time_yday"] = s.shift(spd - h)
    cal[f"{target}_same_time_lastweek"] = s.shift(7 * spd - h)
    return cal


def make_xy(df: pd.DataFrame, target: str, h: int, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.concat([base_features(df, cfg), horizon_features(df, target, h, cfg)], axis=1)
    y = df[target].shift(-h)
    return X, y
