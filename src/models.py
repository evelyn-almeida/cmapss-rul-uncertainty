import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import N_FOLDS, RUL_CAP, SEED


def quantile_model(q, n_estimators):
    return LGBMRegressor(
        objective="quantile", alpha=q, n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        # Start from zero: the capped target puts ~40% of the mass at RUL_CAP,
        # which freezes upper quantiles if boosting starts from the data quantile.
        boost_from_average=False, random_state=SEED, verbose=-1,
    )


def ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


class ConformalQuantileRegressor:
    """Conformalized Quantile Regression (Romano et al., 2019) with cross-validated scores.

    Two quantile models give a raw interval. Out-of-fold errors, computed with
    engine-grouped folds, define a margin that widens (or shrinks) the interval
    so its empirical coverage matches the target on engines never seen in training.
    """

    def __init__(self, coverage, n_estimators):
        self.coverage = coverage
        self.n_estimators = n_estimators
        self.alpha = 1 - coverage
        self.margin = 0.0

    def _fit_pair(self, X, y):
        lower = quantile_model(self.alpha / 2, self.n_estimators).fit(X, y)
        upper = quantile_model(1 - self.alpha / 2, self.n_estimators).fit(X, y)
        return lower, upper

    def fit(self, X, y, groups):
        y = np.asarray(y)
        scores = np.empty(len(y))
        for train_idx, val_idx in GroupKFold(N_FOLDS).split(X, y, groups):
            lower, upper = self._fit_pair(X.iloc[train_idx], y[train_idx])
            lo, hi = lower.predict(X.iloc[val_idx]), upper.predict(X.iloc[val_idx])
            scores[val_idx] = np.maximum(lo - y[val_idx], y[val_idx] - hi)

        level = min(1.0, np.ceil((len(scores) + 1) * self.coverage) / len(scores))
        self.margin = float(np.quantile(scores, level, method="higher"))
        self.lower, self.upper = self._fit_pair(X, y)
        return self

    def predict_raw(self, X):
        lo, hi = self.lower.predict(X), self.upper.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def predict(self, X):
        lo, hi = self.predict_raw(X)
        return np.clip(lo - self.margin, 0, RUL_CAP), np.clip(hi + self.margin, 0, RUL_CAP)
