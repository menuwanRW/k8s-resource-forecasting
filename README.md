# k8s-forecast

Forecasts CPU and memory demand for a Kubernetes workload, then turns the forecast into
**pod** and **node** recommendations. It also flags abnormal pod behavior.

```
Prometheus ──> features ──> quantile forecaster ──> capacity planner ──> pods + nodes
                                   │
                                   └──> anomaly detector (actual vs expected)
```

## How it works

| Piece | File | What it does |
|---|---|---|
| Data | `prometheus.py`, `synthetic.py`, `io.py` | Pulls CPU, memory, pod count and request rate from Prometheus (chunked), or generates synthetic data for a dry run |
| Features | `features.py` | Lags, rolling stats, trend, calendar of the *target* time, and the value at the same time yesterday and last week. Tested to never look ahead |
| Model | `models.py` | One gradient boosting model per target, horizon and quantile. **p50** is the expected load, **p90** is the capacity bound |
| Planner | `scaling.py` | Pods = the HPA formula on the p90 peak over the node lead time. Nodes = bin packing on CPU, memory and pod slots with headroom. Scale up at once, scale down only after a cooldown and in steps |
| Anomalies | `anomaly.py` | Robust z-score of actual usage against the one-step forecast; needs several breaches in a row |
| Evaluation | `evaluate.py` | Accuracy against "stays as now" and "same time yesterday" baselines, plus a scaling backtest |

Sizing from p90 rather than the average is the point: nodes take minutes to arrive, so the
planner provisions for the busy case it expects, instead of reacting after load has landed.

## Quick start

```bash
pip install -e ".[dev]"

# 1. data: synthetic for a dry run, or Prometheus for the real thing
k8sforecast generate --days 28 --out data/synthetic.csv
# k8sforecast --config config.yaml fetch --out data/metrics.csv

# 2. train and see held-out accuracy
k8sforecast --config config.yaml train --data data/synthetic.csv

# 3. scaling backtest: forecast policy vs reactive policy
k8sforecast --config config.yaml evaluate --data data/synthetic.csv

# 4. production: refit on everything, then recommend
k8sforecast --config config.yaml train --data data/metrics.csv --full
k8sforecast --config config.yaml forecast --data data/metrics.csv --current-nodes 3
k8sforecast --config config.yaml anomalies --data data/metrics.csv

pytest
```

Set `pod_cpu_request`, `pod_mem_request_gib` and the node sizes in `config.yaml` to match your
Deployment and node group, otherwise the pod and node counts will not mean anything.

## Results on the synthetic workload (4 held-out days)

| Horizon | CPU error (model) | "stays as now" | "same time yesterday" |
|---|---|---|---|
| 5 min | 0.100 cores | 0.107 | 0.424 |
| 30 min | 0.218 | 0.271 | 0.424 |
| 60 min | 0.269 | 0.430 | 0.426 |

| Policy | Under-provisioned | Average pods |
|---|---|---|
| Reactive (size from current load) | 25.3% of steps | 10.2 |
| Forecast p90 | 5.9% of steps | 10.8 |

These are synthetic numbers, useful for checking the pipeline, not a promise about your
traffic. Run it on your own metrics before trusting any of it.

## Known limitations

- **p90 under-covers when load grows.** Tree models cannot predict above what they saw in
  training, so on a rising trend the p90 bound lands at roughly 73 to 86% coverage instead of
  90%. Retrain often, keep a safety margin in the planner, or add conformal calibration.
- **Anomaly alarms are not free.** About 1% of fault-free held-out steps are flagged, mostly
  load beyond the training range. Route alerts through a duration (for example Alertmanager
  `for: 10m`) before paging anyone.
- **`train --full` evaluates in-sample.** `evaluate` prints a warning when that is the case.
- **One workload per model.** Run it once per Deployment.
- The planner recommends; it does not touch the cluster.

## Putting it into production

1. Run `fetch`, `train --full` and `forecast` as a CronJob every 5 to 15 minutes, retraining daily.
2. Expose the recommendation as a metric (or write it to a ConfigMap) and feed it to KEDA, an HPA
   external metric, or Karpenter / Cluster Autoscaler node-group sizing.
3. Keep your normal HPA as the safety net; use the forecast to raise the floor ahead of load.
