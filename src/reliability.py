import numpy as np
from scipy import stats
from scipy.special import gamma, gammaincc

from config import LEAD_TIME, RUL_CAP


class WeibullFleetModel:
    """Population-level baseline: no sensors, only the age of the engine."""

    def fit(self, lifetimes):
        self.beta, _, self.eta = stats.weibull_min.fit(lifetimes, floc=0)
        return self

    def reliability(self, t):
        return np.exp(-(np.asarray(t) / self.eta) ** self.beta)

    def pdf(self, t):
        return stats.weibull_min.pdf(t, self.beta, scale=self.eta)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1)

    def mean_residual_life(self, t):
        """E[T - t | T > t], closed form via the regularized upper incomplete gamma."""
        z = (np.asarray(t, dtype=float) / self.eta) ** self.beta
        tail = self.eta * gamma(1 + 1 / self.beta) * gammaincc(1 / self.beta, z)
        return tail / np.maximum(np.exp(-z), 1e-300)

    def predict(self, cycles):
        return np.clip(self.mean_residual_life(cycles), 0, RUL_CAP)


def maintenance_tradeoff(trajectories, thresholds, lead_time=LEAD_TIME):
    """Simulate a 'maintain when signal <= threshold' policy on run-to-failure engines.

    trajectories: list of (true_rul, signal) arrays, one pair per engine.
    Returns, per threshold, the share of engines that fail before maintenance
    and the mean useful life thrown away by the engines that were saved.
    """
    failure_rate, wasted_life = [], []
    for h in thresholds:
        failures, wasted = 0, []
        for true_rul, signal in trajectories:
            trigger = np.flatnonzero(signal <= h)
            if len(trigger) == 0 or true_rul[trigger[0]] < lead_time:
                failures += 1
            else:
                wasted.append(true_rul[trigger[0]] - lead_time)
        failure_rate.append(failures / len(trajectories))
        wasted_life.append(np.mean(wasted) if wasted else np.nan)
    return np.array(failure_rate), np.array(wasted_life)
