from __future__ import annotations

import numpy as np
import pandas as pd

from .models import QuantileForecaster


def detect(model: QuantileForecaster, df: pd.DataFrame, target: str = "cpu_cores",
           window: int = 288, z_threshold: float = 5.0, min_run: int = 4) -> pd.DataFrame:
    """Flag abnormal pod behavior: actual usage that departs from the one-step-ahead forecast.

    A robust z-score (median and MAD over a rolling window) keeps normal bursts from tripping it,
    and `min_run` consecutive breaches are required so single noisy scrapes are ignored.
    """
    pred = model.predict_frame(df, target, 1)["q0.5"].shift(1)  # forecast made one step earlier
    resid = df[target] - pred
    med = resid.rolling(window, min_periods=window // 4).median()
    mad = (resid - med).abs().rolling(window, min_periods=window // 4).median()
    z = 0.6745 * (resid - med) / mad.replace(0, np.nan)
    breach = z.abs() > z_threshold
    run = breach.astype(int).groupby((~breach).cumsum()).cumsum()
    return pd.DataFrame({"actual": df[target], "expected": pred, "z": z, "anomaly": run >= min_run})
