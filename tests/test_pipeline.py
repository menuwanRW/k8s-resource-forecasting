import numpy as np

from k8sforecast.anomaly import detect
from k8sforecast.config import Config, ModelConfig
from k8sforecast.evaluate import backtest_scaling, evaluate_models
from k8sforecast.features import make_xy
from k8sforecast.models import QuantileForecaster
from k8sforecast.scaling import CapacityPlanner
from k8sforecast.synthetic import generate


def small_cfg():
    return Config(model=ModelConfig(horizons=[1, 6], max_iter=40))


def test_features_do_not_look_ahead():
    cfg = small_cfg()
    df = generate(days=10)
    X1, _ = make_xy(df, "cpu_cores", 6, cfg)
    t = 2000
    df2 = df.copy()
    df2.iloc[t + 1:] = df2.iloc[t + 1:] * 5 + 100  # rewrite the future
    X2, _ = make_xy(df2, "cpu_cores", 6, cfg)
    a, b = X1.iloc[t].to_numpy(), X2.iloc[t].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_forecast_shape_and_quantile_order():
    cfg = small_cfg()
    df = generate(days=12)
    model = QuantileForecaster(cfg).fit(df.iloc[:-300])
    fc = model.forecast_latest(df)
    assert list(fc.index) == [1, 6]
    assert (fc["cpu_cores_q0.9"] >= fc["cpu_cores_q0.5"]).all()
    assert (fc["mem_gib_q0.9"] >= fc["mem_gib_q0.5"]).all()


def test_model_beats_persistence_at_30_minutes():
    cfg = Config(model=ModelConfig(horizons=[1, 6], max_iter=150))
    df = generate(days=28)
    cut = df.index[int(len(df) * 0.8)]
    model = QuantileForecaster(cfg).fit(df[df.index < cut])
    table = evaluate_models(model, df, cut)
    row = table[(table.target == "cpu_cores") & (table.horizon_min == 30)].iloc[0]
    assert row["mae"] < row["mae_persistence"]


def test_planner_scale_up_and_cooldown():
    import pandas as pd
    planner = CapacityPlanner(Config().scaling)
    high = pd.DataFrame({"cpu_cores_q0.5": [4.0], "cpu_cores_q0.9": [5.0],
                         "mem_gib_q0.5": [4.0], "mem_gib_q0.9": [5.0]}, index=[1])
    up = planner.recommend(high, current_pods=4, current_nodes=2)
    assert up.pod_action == "scale up" and up.pods >= 15
    low = high * 0.1
    held = planner.recommend(low, current_pods=20, current_nodes=3, steps_since_change=2)
    assert held.pods == 20 and "cooldown" in held.pod_action
    down = planner.recommend(low, current_pods=20, current_nodes=3, steps_since_change=100)
    assert 15 <= down.pods < 20  # stepwise, not a cliff


def test_forecast_policy_reduces_shortfall():
    cfg = small_cfg()
    df = generate(days=21)
    cut = df.index[int(len(df) * 0.8)]
    model = QuantileForecaster(cfg).fit(df[df.index < cut])
    bt = backtest_scaling(model, df, cut).set_index("policy")
    assert bt.loc["forecast_p90", "under_provisioned_%"] < bt.loc["reactive", "under_provisioned_%"]


def test_anomaly_detector_flags_injected_fault():
    cfg = Config(model=ModelConfig(horizons=[1, 6], max_iter=150))
    df = generate(days=21)
    held = 1200  # last four days are never seen in training
    model = QuantileForecaster(cfg).fit(df.iloc[:-held])
    bad = df.copy()
    i = len(bad) - 200
    bad.iloc[i:i + 12, bad.columns.get_loc("cpu_cores")] += 3.0  # runaway thread: CPU up, traffic flat
    res = detect(model, bad)
    assert res["anomaly"].iloc[i:i + 12].any()
    clean = res["anomaly"].iloc[-held:i - 50]
    assert clean.mean() < 0.03  # low false-alarm rate on unseen, fault-free data
