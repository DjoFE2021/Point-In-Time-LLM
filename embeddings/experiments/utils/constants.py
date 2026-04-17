import numpy as np

# Default grid of shrinkage values for ridge regression.
# When normalize_by_trace=True (default in Ridge), these are dimensionless
# ratios relative to trace(S'S/t), so the grid is scale-invariant across
# datasets with different signal magnitudes.
DEFAULT_SHRINKAGE_GRID: np.ndarray = np.array([10**i for i in range(-6,7)])
