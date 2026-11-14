from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .anomaly import detect
from .config import Config
from .evaluate import backtest_scaling, evaluate_models
from .io import load_metrics, split_point
from .models import QuantileForecaster
from .prometheus import fetch_workload
from .scaling import CapacityPlanner
from .synthetic import generate

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _save_csv(df: pd.DataFrame, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"wrote {len(df)} rows to {out}")


def cmd_generate(a, cfg):
    _save_csv(generate(days=a.days, step_seconds=cfg.data.step_seconds, seed=a.seed), a.out)


def cmd_fetch(a, cfg):
    _save_csv(fetch_workload(cfg), a.out)


def cmd_train(a, cfg):
    df = load_metrics(a.data, cfg)
    cut = split_point(df, cfg.model.test_fraction)
    model = QuantileForecaster(cfg).fit(df[df.index < cut])
    print("Held-out accuracy (test period starts", cut, ")")
    print(evaluate_models(model, df, cut).to_string(index=False))
    if a.full:
        model = QuantileForecaster(cfg).fit(df)  # refit on everything for production use
    model.save(a.model)
    print("saved model to", a.model)


def cmd_evaluate(a, cfg):
    df = load_metrics(a.data, cfg)
    model = QuantileForecaster.load(a.model)
    cut = split_point(df, cfg.model.test_fraction)
    trained_until = getattr(model, "trained_until", None)
    if trained_until is not None and trained_until >= cut:
        print("WARNING: this model was trained on the test period (train --full), so the numbers "
              "below are in-sample and too optimistic. Train without --full to get honest scores.\n")
    print(evaluate_models(model, df, cut).to_string(index=False))
    print("\nScaling backtest")
    print(backtest_scaling(model, df, cut).to_string(index=False))


def cmd_forecast(a, cfg):
    df = load_metrics(a.data, cfg)
    model = QuantileForecaster.load(a.model)
    fc = model.forecast_latest(df)
    if a.current_pods is not None:
        pods = a.current_pods
    elif "pods" in df.columns:
        pods = int(round(df["pods"].iloc[-1]))
    else:
        pods = cfg.scaling.min_pods
    rec = CapacityPlanner(cfg.scaling).recommend(fc, pods, a.current_nodes, a.steps_since_change)
    print(fc.round(3).to_string())
    print(json.dumps(rec.to_dict(), indent=2))


def cmd_anomalies(a, cfg):
    df = load_metrics(a.data, cfg)
    model = QuantileForecaster.load(a.model)
    res = detect(model, df)
    print(f"{int(res['anomaly'].sum())} anomalous steps")
    print(res[res["anomaly"]].tail(a.last).round(3).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="k8sforecast")
    p.add_argument("--config", default=None, help="YAML config (defaults are used if omitted)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate", help="create synthetic metrics for a dry run")
    s.add_argument("--days", type=int, default=28)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/synthetic.csv")
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("fetch", help="pull metrics from Prometheus")
    s.add_argument("--out", default="data/metrics.csv")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("train")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/forecaster.joblib")
    s.add_argument("--full", action="store_true", help="refit on all data after evaluating")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/forecaster.joblib")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("forecast", help="forecast and recommend pods and nodes now")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/forecaster.joblib")
    s.add_argument("--current-pods", type=int, default=None)
    s.add_argument("--current-nodes", type=int, required=True)
    s.add_argument("--steps-since-change", type=int, default=10**9)
    s.set_defaults(fn=cmd_forecast)

    s = sub.add_parser("anomalies", help="flag abnormal pod behavior")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/forecaster.joblib")
    s.add_argument("--last", type=int, default=20)
    s.set_defaults(fn=cmd_anomalies)

    a = p.parse_args(argv)
    a.fn(a, Config.from_yaml(a.config))


if __name__ == "__main__":
    sys.exit(main())
