"""End-to-end pipeline: python run.py [--quick]"""
import argparse
import time

import numpy as np
import pandas as pd

from config import COVERAGE, COVERAGE_GRID, MODES, RESULTS_DIR
from src import plots
from src.data import N_CONDITIONS, last_cycle, load_subset, normalize_by_condition, split_units
from src.features import build_features, feature_columns
from src.metrics import coverage, mean_width, nasa_score, rmse
from src.models import ConformalQuantileRegressor, quantile_model, ridge_model
from src.reliability import WeibullFleetModel, maintenance_tradeoff


def run_subset(name, n_estimators):
    raw_train, raw_test = load_subset(name)
    norm_train, norm_test = normalize_by_condition(raw_train, raw_test, N_CONDITIONS[name])
    train, test = build_features(norm_train), build_features(norm_test)
    cols = feature_columns(train)
    fit, hold = split_units(train)
    test = last_cycle(test)

    median = quantile_model(0.5, n_estimators).fit(fit[cols], fit["rul"])
    ridge = ridge_model().fit(fit[cols], fit["rul"])
    lifetimes = fit.groupby("unit")["cycle"].max().to_numpy()
    weibull = WeibullFleetModel().fit(lifetimes)
    intervals = {c: ConformalQuantileRegressor(c, n_estimators).fit(fit[cols], fit["rul"], fit["unit"])
                 for c in COVERAGE_GRID}

    def predict(df):
        lo, hi = intervals[COVERAGE].predict(df[cols])
        return pd.DataFrame({
            "unit": df["unit"], "cycle": df["cycle"], "y": df["rul"], "rul_true": df["rul_true"],
            "pred": np.clip(median.predict(df[cols]), 0, None), "lo": lo, "hi": hi,
            "ridge": np.clip(ridge.predict(df[cols]), 0, None),
            "weibull": weibull.predict(df["cycle"]),
        })

    t, h = predict(test), predict(hold)
    metrics = {
        "Test engines": len(t),
        "RMSE Weibull": rmse(t["y"], t["weibull"]),
        "RMSE Ridge": rmse(t["y"], t["ridge"]),
        "RMSE LightGBM": rmse(t["y"], t["pred"]),
        "NASA score LightGBM": nasa_score(t["y"], t["pred"]),
        f"Coverage {COVERAGE:.0%} (raw)": coverage(t["y"], *intervals[COVERAGE].predict_raw(test[cols])),
        f"Coverage {COVERAGE:.0%} (conformal)": coverage(t["y"], t["lo"], t["hi"]),
        "Mean interval width": mean_width(t["lo"], t["hi"]),
    }
    calibration = [{"subset": name, "coverage": c,
                    "raw": coverage(hold["rul"], *m.predict_raw(hold[cols])),
                    "conformal": coverage(hold["rul"], *m.predict(hold[cols]))}
                   for c, m in intervals.items()]

    return {"metrics": metrics, "test": {k: t[k].to_numpy() for k in t}, "holdout": h,
            "calibration": calibration, "raw_train": raw_train, "norm_train": norm_train,
            "lifetimes": lifetimes, "weibull": weibull, "median_model": median, "columns": cols}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="FD001 only, fewer trees")
    mode = MODES["quick" if parser.parse_args().quick else "full"]

    results = {}
    for name in mode["subsets"]:
        start = time.time()
        results[name] = run_subset(name, mode["n_estimators"])
        print(f"{name} done in {time.time() - start:.0f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = pd.DataFrame({k: r["metrics"] for k, r in results.items()}).T
    calibration = pd.DataFrame([row for r in results.values() for row in r["calibration"]])
    metrics.to_csv(RESULTS_DIR / "metrics.csv", float_format="%.4f")
    calibration.to_csv(RESULTS_DIR / "calibration.csv", index=False, float_format="%.4f")
    (RESULTS_DIR / "metrics.md").write_text(metrics.round(3).to_markdown())

    holdout = pd.concat([r["holdout"] for r in results.values()], keys=results.keys(), names=["subset"])
    holdout.reset_index(level=0).to_csv(RESULTS_DIR / "holdout_predictions.csv", index=False, float_format="%.2f")
    engines = [g for _, g in holdout.groupby(["subset", "unit"])]
    thresholds = np.arange(0, 126, 3)
    tradeoffs = {
        "Age-based (Weibull)": maintenance_tradeoff(
            [(g["rul_true"].to_numpy(), g["weibull"].to_numpy()) for g in engines], thresholds),
        "Point forecast": maintenance_tradeoff(
            [(g["rul_true"].to_numpy(), g["pred"].to_numpy()) for g in engines], thresholds),
        f"Lower {COVERAGE:.0%} bound": maintenance_tradeoff(
            [(g["rul_true"].to_numpy(), g["lo"].to_numpy()) for g in engines], thresholds),
    }

    first = next(iter(results))
    multi = "FD002" if "FD002" in results else first
    r0 = results[first]
    plots.setup_style()
    plots.sensor_trajectories(r0["raw_train"], first)
    plots.sensor_correlation(r0["norm_train"], first)
    plots.operating_conditions(results[multi]["raw_train"], results[multi]["norm_train"], multi)
    plots.fleet_weibull(r0["lifetimes"], r0["weibull"], first)
    plots.predicted_vs_true(results)
    plots.test_intervals_sorted(r0, first)
    plots.rul_trajectories(results)
    plots.calibration_curve(calibration)
    plots.error_distribution(results)
    plots.feature_importance(r0["median_model"], r0["columns"], first)
    plots.maintenance_policy(tradeoffs, len(engines))
    plots.model_comparison(metrics)

    print("\n" + metrics.round(3).to_string())
    print("\n" + calibration.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
