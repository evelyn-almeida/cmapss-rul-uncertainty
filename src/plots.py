import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import COVERAGE, FIG_DIR, LEAD_TIME, RUL_CAP
from src.data import SENSOR_NAMES, SENSORS

BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#e07a1f", "#2a9d8f", "#c0392b", "#8a8a8a"
POLICY_COLORS = {"Age-based (Weibull)": GRAY, "Point forecast": BLUE,
                 f"Lower {COVERAGE:.0%} bound": ORANGE}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _grid(n):
    rows, cols = (1, 1) if n == 1 else (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False)
    return fig, axes.ravel()


def sensor_trajectories(raw_train, subset):
    sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, s in zip(axes.ravel(), sensors):
        for _, g in raw_train.groupby("unit"):
            ax.plot(-g["rul_true"], g[s].rolling(5, min_periods=1).mean(), color=BLUE, alpha=0.12, lw=0.8)
        fleet = raw_train[raw_train["rul_true"] <= 250].groupby("rul_true")[s].median()
        ax.plot(-fleet.index, fleet.rolling(10, min_periods=1).mean(), color=ORANGE, lw=2.5, label="Fleet median")
        ax.set_title(f"{s} · {SENSOR_NAMES[s]}")
        ax.set_xlabel("Cycles before failure")
        ax.set_xlim(-300, 0)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"{subset}: every engine drifts toward failure, but each one at its own pace",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_sensor_trajectories")


def sensor_correlation(train, subset):
    corr = pd.Series({s: spearmanr(train[s], train["rul_true"])[0] for s in SENSORS})
    corr = corr.reindex(corr.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([f"{s} · {SENSOR_NAMES[s]}" for s in corr.index], corr,
            color=[ORANGE if v < 0 else BLUE for v in corr])
    ax.axvline(0, color="black", lw=0.8)
    ax.text(0.02, 0.02, "◀ rises as failure approaches", color=ORANGE, transform=ax.transAxes)
    ax.text(0.98, 0.02, "falls as failure approaches ▶", color=BLUE, ha="right", transform=ax.transAxes)
    ax.set_xlabel("Spearman correlation with remaining useful life")
    ax.set_title(f"{subset}: which sensors carry the degradation signal?")
    save(fig, "02_sensor_correlation")


def operating_conditions(raw_train, norm_train, subset, unit=1):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    sample = norm_train.sample(min(5000, len(norm_train)), random_state=0)
    raw_sample = raw_train.loc[sample.index]
    axes[0].scatter(raw_sample["setting_1"], raw_sample["setting_2"], c=sample["condition"],
                    cmap="tab10", s=90, alpha=0.8, edgecolor="white")
    axes[0].set(title="Six flight regimes found by K-means", xlabel="Altitude setting", ylabel="Mach setting")

    raw = raw_train[raw_train["unit"] == unit]
    norm = norm_train[norm_train["unit"] == unit]
    axes[1].plot(raw["cycle"], raw["s2"], color=GRAY, lw=0.8)
    axes[1].set(title=f"Raw sensor s2, engine {unit}", xlabel="Cycle", ylabel="Temperature (°R)")
    axes[2].plot(norm["cycle"], norm["s2"], color=BLUE, lw=0.6, alpha=0.5)
    axes[2].plot(norm["cycle"], norm["s2"].rolling(15, min_periods=1).mean(), color=ORANGE, lw=2.2)
    axes[2].set(title="After per-regime normalization", xlabel="Cycle", ylabel="z-score")
    fig.suptitle(f"{subset}: removing the operating condition reveals the hidden degradation trend",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_operating_conditions")


def fleet_weibull(lifetimes, weibull, subset):
    t = np.linspace(1, lifetimes.max() * 1.15, 400)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].hist(lifetimes, bins=20, density=True, color=BLUE, alpha=0.45, edgecolor="white")
    axes[0].plot(t, weibull.pdf(t), color=ORANGE, lw=2.5)
    axes[0].set(title="Lifetime distribution", xlabel="Cycles to failure", ylabel="Density")

    ordered = np.sort(lifetimes)
    empirical = 1 - np.arange(1, len(ordered) + 1) / len(ordered)
    axes[1].step(ordered, empirical, where="post", color=BLUE, label="Empirical")
    axes[1].plot(t, weibull.reliability(t), color=ORANGE, lw=2.5, label="Weibull fit")
    axes[1].set(title="Reliability R(t)", xlabel="Cycles", ylabel="P(engine still running)")
    axes[1].legend()

    axes[2].plot(t, weibull.hazard(t), color=RED, lw=2.5)
    axes[2].set(title="Hazard rate h(t)", xlabel="Cycles", ylabel="Instantaneous failure rate")
    axes[2].text(0.05, 0.9, f"β = {weibull.beta:.2f}  (β > 1: wear-out)\nη = {weibull.eta:.0f} cycles",
                 transform=axes[2].transAxes, va="top",
                 bbox=dict(boxstyle="round", fc="white", ec=GRAY))
    fig.suptitle(f"{subset}: classical reliability view of the fleet (no sensors)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_fleet_weibull")


def predicted_vs_true(results):
    fig, axes = _grid(len(results))
    for ax, (name, r) in zip(axes, results.items()):
        t = r["test"]
        covered = (t["y"] >= t["lo"]) & (t["y"] <= t["hi"])
        ax.vlines(t["y"], t["lo"], t["hi"], color=GRAY, alpha=0.35, lw=1)
        ax.scatter(t["y"][covered], t["pred"][covered], color=BLUE, s=18, label="Inside interval")
        ax.scatter(t["y"][~covered], t["pred"][~covered], color=RED, s=22, label="Outside interval")
        ax.plot([0, RUL_CAP], [0, RUL_CAP], color="black", ls="--", lw=1)
        ax.set(title=name, xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)",
               xlim=(-5, RUL_CAP + 10), ylim=(-5, RUL_CAP + 10))
        ax.text(0.04, 0.95, f"RMSE = {r['metrics']['RMSE LightGBM']:.1f}\n"
                            f"Coverage = {r['metrics'][f'Coverage {COVERAGE:.0%} (conformal)']:.0%}",
                transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", fc="white", ec=GRAY))
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Test set: median forecast with {COVERAGE:.0%} conformal intervals",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_predicted_vs_true")


def test_intervals_sorted(result, subset):
    t = result["test"]
    order = np.argsort(t["y"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(x, t["lo"][order], t["hi"][order], color=BLUE, alpha=0.2,
                    label=f"{COVERAGE:.0%} prediction interval")
    ax.scatter(x, t["pred"][order], color=BLUE, s=14, label="Predicted (median)")
    ax.plot(x, t["y"][order], color=ORANGE, lw=2.5, label="True RUL")
    ax.set(xlabel="Test engines, sorted by true RUL", ylabel="RUL (cycles)",
           title=f"{subset}: one forecast per test engine — the band tells us how much to trust it")
    ax.legend(loc="upper left")
    save(fig, "06_test_intervals_sorted")


def rul_trajectories(results):
    panels = []
    if len(results) >= 4:
        for name, r in results.items():
            panels.append((name, r["holdout"], _typical_unit(r["holdout"])))
    else:
        name, r = next(iter(results.items()))
        units = r["holdout"].groupby("unit")["cycle"].max().sort_values().index
        for u in units[np.linspace(0, len(units) - 1, 4).astype(int)]:
            panels.append((name, r["holdout"], u))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (name, h, unit) in zip(axes.ravel(), panels):
        e = h[h["unit"] == unit]
        ax.fill_between(e["cycle"], e["lo"], e["hi"], color=BLUE, alpha=0.2, label=f"{COVERAGE:.0%} interval")
        ax.plot(e["cycle"], e["pred"], color=BLUE, lw=2, label="Predicted RUL")
        ax.plot(e["cycle"], e["y"], color=ORANGE, lw=2, ls="--", label="True RUL (capped)")
        ax.plot(e["cycle"], e["rul_true"], color=GRAY, lw=1, ls=":", label="True RUL (raw)")
        ax.set(title=f"{name} · engine {unit} (never seen in training)", xlabel="Cycle",
               ylabel="RUL (cycles)", ylim=(0, RUL_CAP + 25))
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Live monitoring: the forecast sharpens as the engine approaches failure",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "07_rul_trajectories")


def _typical_unit(df):
    life = df.groupby("unit")["cycle"].max()
    return (life - life.median()).abs().idxmin()


def calibration_curve(calibration):
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0.4, 1], [0.4, 1], color="black", ls="--", lw=1, label="Perfect calibration")
    for col, color, label in [("raw", RED, "Quantile regression only"),
                              ("conformal", GREEN, "+ conformal calibration")]:
        ax.scatter(calibration["coverage"], calibration[col], color=color, alpha=0.35, s=25)
        mean = calibration.groupby("coverage")[col].mean()
        ax.plot(mean.index, mean.values, color=color, lw=2.5, marker="o", label=label)
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage (holdout engines)",
           title="Are the intervals honest?", xlim=(0.4, 1), ylim=(0.4, 1))
    ax.legend(loc="lower right")
    save(fig, "08_calibration_curve")


def error_distribution(results):
    errors = {k: np.concatenate([r["test"][k] - r["test"]["y"] for r in results.values()])
              for k in ["pred", "ridge", "weibull"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-80, 80, 41)
    for key, color, label in [("weibull", GRAY, "Weibull (age only)"),
                              ("ridge", RED, "Ridge regression"),
                              ("pred", BLUE, "LightGBM (median)")]:
        axes[0].hist(errors[key], bins=bins, color=color, alpha=0.45, label=label)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set(title="Prediction error on all test engines", xlabel="Predicted − true RUL (cycles)",
                ylabel="Engines")
    axes[0].legend()

    d = np.linspace(-40, 40, 400)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    axes[1].plot(d, score, color="black", lw=2)
    axes[1].fill_between(d, score, where=d > 0, color=RED, alpha=0.25, label="Late: engine may fail first")
    axes[1].fill_between(d, score, where=d < 0, color=GREEN, alpha=0.25, label="Early: wasted useful life")
    axes[1].set(title="Why late predictions cost more (PHM08 score)", xlabel="Predicted − true RUL (cycles)",
                ylabel="Penalty")
    axes[1].legend(loc="upper center")
    fig.tight_layout()
    save(fig, "09_error_distribution")


def feature_importance(model, columns, subset, top=15):
    imp = pd.Series(model.booster_.feature_importance("gain"), index=columns)
    imp = (imp / imp.sum()).sort_values().tail(top)
    labels = [f"{c}  ({SENSOR_NAMES.get(c.split('_')[0], 'elapsed cycles')})" for c in imp.index]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(labels, imp.values * 100, color=BLUE)
    ax.set(xlabel="Share of total gain (%)", title=f"{subset}: what the model looks at")
    save(fig, "10_feature_importance")


def maintenance_policy(tradeoffs, n_engines, target=0.05):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    lines = []
    for name, (fail, wasted) in tradeoffs.items():
        ax.plot(wasted, fail * 100, color=POLICY_COLORS[name], lw=2.5, marker="o", ms=3, label=name)
        ok = fail <= target
        if ok.any():
            best = np.nanmin(wasted[ok])
            ax.scatter(best, fail[ok][np.nanargmin(wasted[ok])] * 100, color=POLICY_COLORS[name],
                       s=110, edgecolor="black", zorder=5)
            lines.append(f"{name}: {best:.0f} cycles")
    ax.axhline(target * 100, color=RED, ls="--", lw=1)
    ax.text(0.98, target * 100 + 1, f"{target:.0%} failure-risk budget", color=RED,
            ha="right", transform=ax.get_yaxis_transform())
    ax.text(0.30, 0.62, f"Wasted life at ≤{target:.0%} failures\n" + "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", fc="white", ec=GRAY))
    ax.set_xscale("log")
    ax.set(xlabel="Mean useful life thrown away per engine (cycles, log scale)",
           ylabel="Engines that fail before maintenance (%)",
           title=f"From forecast to decision: maintenance policies on {n_engines} unseen engines "
                 f"(lead time = {LEAD_TIME} cycles)")
    ax.legend(loc="center right")
    ax.set_xlim(left=1)
    save(fig, "11_maintenance_policy")


def model_comparison(metrics):
    models = [("RMSE Weibull", GRAY, "Weibull (age only)"), ("RMSE Ridge", RED, "Ridge regression"),
              ("RMSE LightGBM", BLUE, "LightGBM (median)")]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (col, color, label) in enumerate(models):
        bars = ax.bar(x + (i - 1) * 0.27, metrics[col], width=0.27, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, metrics.index)
    ax.set(ylabel="Test RMSE (cycles, lower is better)", title="Sensors + ML vs. the fleet-average model")
    ax.legend()
    save(fig, "12_model_comparison")
