import numpy as np


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def nasa_score(y, pred):
    """PHM08 asymmetric score: late predictions (pred > y) are penalized harder."""
    d = np.asarray(pred) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi):
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
