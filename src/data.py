import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import DATA_DIR, HOLDOUT_FRACTION, RUL_CAP, SEED

SETTINGS = ["setting_1", "setting_2", "setting_3"]
COLUMNS = ["unit", "cycle"] + SETTINGS + [f"s{i}" for i in range(1, 22)]

# Sensors that carry degradation signal (the other seven are constant or near-constant)
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
           "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

SENSOR_NAMES = {
    "s2": "LPC outlet temp.", "s3": "HPC outlet temp.", "s4": "LPT outlet temp.",
    "s7": "HPC outlet pressure", "s8": "Fan speed", "s9": "Core speed",
    "s11": "HPC static pressure", "s12": "Fuel flow / Ps30", "s13": "Corrected fan speed",
    "s14": "Corrected core speed", "s15": "Bypass ratio", "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}

N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_subset(name):
    train = _read(DATA_DIR / f"train_{name}.txt")
    test = _read(DATA_DIR / f"test_{name}.txt")
    rul_at_end = pd.read_csv(DATA_DIR / f"RUL_{name}.txt", header=None).iloc[:, 0].to_numpy()

    train["rul_true"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    test["rul_true"] = rul_at_end[test["unit"] - 1] + last_cycle - test["cycle"]

    for df in (train, test):
        df["rul"] = df["rul_true"].clip(upper=RUL_CAP)
    return train, test


def normalize_by_condition(train, test, n_conditions):
    """Cluster operating conditions and z-score each sensor within its condition."""
    train, test = train.copy(), test.copy()
    kmeans = KMeans(n_clusters=n_conditions, n_init=10, random_state=SEED).fit(train[SETTINGS])
    train["condition"] = kmeans.predict(train[SETTINGS])
    test["condition"] = kmeans.predict(test[SETTINGS])

    mean = train.groupby("condition")[SENSORS].mean()
    std = train.groupby("condition")[SENSORS].std().replace(0, 1).fillna(1)
    for df in (train, test):
        c = df["condition"]
        df[SENSORS] = (df[SENSORS] - mean.loc[c].to_numpy()) / std.loc[c].to_numpy()
    return train, test


def split_units(df):
    """Split engines (not rows) so that no engine appears on both sides."""
    units = df["unit"].unique()
    rng = np.random.default_rng(SEED)
    holdout = rng.choice(units, size=int(HOLDOUT_FRACTION * len(units)), replace=False)
    mask = df["unit"].isin(holdout)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def last_cycle(df):
    return df.groupby("unit").tail(1).reset_index(drop=True)
