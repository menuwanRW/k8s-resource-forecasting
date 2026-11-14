from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Config
from .features import make_xy


class QuantileForecaster:
    """Direct multi-horizon quantile forecaster.

    One gradient boosting model per (target, horizon, quantile). p50 is the expected value and
    p90 is the capacity bound used for scaling decisions, so the planner sizes for the busy case
    without waiting for the load to arrive.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.quantiles = sorted(cfg.model.quantiles)
        self.models: dict[tuple[str, int, float], HistGradientBoostingRegressor] = {}
        self.columns: dict[tuple[str, int], list[str]] = {}
        self.trained_until: pd.Timestamp | None = None

    def fit(self, df: pd.DataFrame) -> "QuantileForecaster":
        m = self.cfg.model
        self.trained_until = df.index[-1]
        for target in m.targets:
            for h in m.horizons:
                X, y = make_xy(df, target, h, self.cfg)
                mask = y.notna()
                X, y = X[mask], y[mask]
                self.columns[(target, h)] = list(X.columns)
                for q in self.quantiles:
                    est = HistGradientBoostingRegressor(
                        loss="quantile", quantile=q, max_iter=m.max_iter,
                        learning_rate=m.learning_rate, max_depth=m.max_depth,
                        l2_regularization=1.0, random_state=0,
                    )
                    self.models[(target, h, q)] = est.fit(X, y)
        return self

    def _predict(self, X: pd.DataFrame, target: str, h: int) -> np.ndarray:
        cols = self.columns[(target, h)]
        preds = np.column_stack([self.models[(target, h, q)].predict(X[cols]) for q in self.quantiles])
        return np.sort(preds, axis=1)  # stop quantile crossing (p90 never below p50)

    def predict_frame(self, df: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
        """Forecast for every row: the value at index + h, indexed by when it was issued."""
        X, _ = make_xy(df, target, h, self.cfg)
        out = self._predict(X, target, h)
        return pd.DataFrame(out, index=df.index, columns=[f"q{q}" for q in self.quantiles])

    def forecast_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forecast from the most recent observation. Rows are horizons in steps; columns look
        like cpu_cores_q0.9."""
        rows = {}
        for h in self.cfg.model.horizons:
            row = {}
            for target in self.cfg.model.targets:
                X, _ = make_xy(df, target, h, self.cfg)
                pred = self._predict(X.iloc[[-1]], target, h)[0]
                for q, v in zip(self.quantiles, pred):
                    row[f"{target}_q{q}"] = float(v)
            rows[h] = row
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "horizon_steps"
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "QuantileForecaster":
        return joblib.load(path)
