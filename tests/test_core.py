import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.metrics import coverage, nasa_score, rmse
from src.models import ConformalQuantileRegressor
from src.reliability import WeibullFleetModel, maintenance_tradeoff


def test_metrics():
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert nasa_score([10], [10]) == 0
    assert nasa_score([10], [15]) > nasa_score([10], [5])  # late is worse than early
    assert coverage([1, 2, 3, 4], [0, 0, 0, 5], [2, 2, 2, 6]) == 0.5


def test_weibull_recovers_parameters():
    lifetimes = 200 * np.random.default_rng(0).weibull(4.0, 5000)
    model = WeibullFleetModel().fit(lifetimes)
    assert abs(model.beta - 4.0) < 0.2
    assert abs(model.eta - 200) < 5
    assert model.mean_residual_life(0) > model.mean_residual_life(150)


def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"x": rng.uniform(0, 100, n)})
    y = np.clip(X["x"] + rng.normal(0, 10, n), 0, 125).to_numpy()
    groups = np.repeat(np.arange(40), n // 40)
    model = ConformalQuantileRegressor(0.9, n_estimators=100).fit(X, y, groups)
    X_new = pd.DataFrame({"x": rng.uniform(0, 100, n)})
    y_new = np.clip(X_new["x"] + rng.normal(0, 10, n), 0, 125)
    assert 0.86 <= coverage(y_new, *model.predict(X_new)) <= 0.95


def test_maintenance_tradeoff():
    true_rul = np.arange(50, -1, -1)
    perfect = [(true_rul, true_rul.astype(float))]
    fail, wasted = maintenance_tradeoff(perfect, thresholds=[5, 20], lead_time=10)
    assert fail.tolist() == [1.0, 0.0]
    assert wasted[1] == 10
