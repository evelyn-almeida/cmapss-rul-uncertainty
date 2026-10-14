import pandas as pd

from src.data import SENSORS

WINDOWS = (5, 15, 30)
ID_COLUMNS = ["unit", "cycle", "condition", "rul", "rul_true"]


def build_features(df):
    """Rolling statistics per engine: level, short-vs-long trend and volatility."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit")[SENSORS]

    def rolling(window, stat, min_periods=1):
        out = getattr(grouped.rolling(window, min_periods=min_periods), stat)()
        return out.reset_index(level=0, drop=True)

    means = {w: rolling(w, "mean") for w in WINDOWS}
    parts = [df[ID_COLUMNS]]
    parts += [means[w].add_suffix(f"_mean{w}") for w in WINDOWS]
    parts.append((means[5] - means[30]).add_suffix("_trend"))
    parts.append(rolling(15, "std", min_periods=2).fillna(0).add_suffix("_std15"))
    return pd.concat(parts, axis=1)


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLUMNS or c == "cycle"]
