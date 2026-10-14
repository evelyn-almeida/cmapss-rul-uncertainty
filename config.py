from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
RUL_CAP = 125            # piecewise-linear target: healthy engines are labeled RUL = 125
COVERAGE = 0.90          # nominal coverage of the main prediction interval
COVERAGE_GRID = (0.50, 0.80, 0.90)
HOLDOUT_FRACTION = 0.15  # engines kept aside for evaluation and policy simulation
N_FOLDS = 5              # cross-conformal calibration folds
LEAD_TIME = 10           # cycles needed to schedule maintenance

MODES = {
    "quick": {"subsets": ["FD001"], "n_estimators": 120},
    "full": {"subsets": ["FD001", "FD002", "FD003", "FD004"], "n_estimators": 300},
}
