"""Shared constants for the embedding portfolio experiments."""
import numpy as np

# Ridge shrinkage grid; with normalize_by_trace it is scaled by trace(S'S/t), so scale-free
DEFAULT_SHRINKAGE_GRID: np.ndarray = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 100.0])