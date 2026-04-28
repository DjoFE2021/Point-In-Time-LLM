import numpy as np

# Default grid of shrinkage values for ridge regression.
# When normalize_by_trace=True (default in Ridge), these are dimensionless
# ratios relative to trace(S'S/t), so the grid is scale-invariant across
# datasets with different signal magnitudes.
DEFAULT_SHRINKAGE_GRID: np.ndarray = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 100.0])