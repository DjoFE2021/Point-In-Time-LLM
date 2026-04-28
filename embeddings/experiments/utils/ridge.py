import numpy as np
from typing import Optional

from .constants import DEFAULT_SHRINKAGE_GRID


class Ridge:
    """
    Ridge regression over a grid of regularization parameters.

    Regression solves:
        beta = (z I + S'S/t)^{-1} S'y/t

    Eigendecomposition of S'S/t (or SS'/t in the wide regime) is computed
    once in fit(), so beta(z) for each z in the grid costs only a diagonal
    scaling — no repeated matrix inversions.

    Dimension-swap trick (p >= t)
    ------------------------------
    By the Woodbury identity:
        (z I_p + S'S/t)^{-1} S' = S' (z I_t + SS'/t)^{-1}

    so we decompose the cheaper t×t matrix SS'/t instead of the p×p matrix S'S/t.

    Trace normalization
    -------------------
    When normalize_by_trace=True, the effective shrinkage applied is:
        z_eff = z * trace(S'S/t)
    where trace(S'S/t) = ||S||²_F / t is the total signal variance.
    This makes the grid entries dimensionless ratios, so the same grid works
    across datasets with different signal magnitudes.

    Parameters
    ----------
    shrinkage_list   : 1-D array of regularization values (default: DEFAULT_SHRINKAGE_GRID)
    normalize_by_trace: if True, scale each z by trace(S'S/t) at fit time
    """

    def __init__(
        self,
        shrinkage_list: np.ndarray = DEFAULT_SHRINKAGE_GRID,
        normalize_by_trace: bool = True,
    ):
        self.shrinkage_list = np.atleast_1d(np.asarray(shrinkage_list, dtype=float))
        self.normalize_by_trace = normalize_by_trace
        self._fitted = False

    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
    ) -> "Ridge":
        """
        Compute betas for every value in shrinkage_list.

        Parameters
        ----------
        signals : array of shape (t, p)
        labels  : array of shape (t,) or (t, k)

        Returns
        -------
        self  (betas stored in self.betas, shape (p, n_z))
        """
        signals = np.asarray(signals, dtype=float)
        labels = np.asarray(labels, dtype=float)

        t_, p_ = signals.shape
        self._t = t_
        self._p = p_
        self._signals = signals

        # Effective shrinkage: optionally scaled by trace(S'S/t) = ||S||²_F / t
        if self.normalize_by_trace:
            trace = np.sum(signals ** 2) / t_
            effective_shrinkage = (p_/t_)*self.shrinkage_list * trace
        else:
            effective_shrinkage = self.shrinkage_list
        self._effective_shrinkage = effective_shrinkage

        if p_ < t_:
            self.betas, self._eig_vecs, self._eig_vals = self._fit_standard(
                signals, labels, t_, effective_shrinkage
            )
            self._wide = False
        else:
            self.betas, self._eig_vecs, self._eig_vals = self._fit_swap(
                signals, labels, t_, effective_shrinkage
            )
            self._wide = True

        self._fitted = True
        return self

    def hat_diagonal(self, signals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the ridge hat-matrix diagonal h_ii for every z in shrinkage_list.

        Uses the training signals stored during fit() if signals is None.

        Returns
        -------
        h : array of shape (t, n_z)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before hat_diagonal().")
        X = self._signals if signals is None else np.asarray(signals, dtype=float)
        t_ = X.shape[0]
        V, d, z = self._eig_vecs, self._eig_vals, self._effective_shrinkage

        if not self._wide:
            # standard case: X'X/t = V D V'
            # h_ii(z_k) = (1/t) * sum_j (V'x_i)_j^2 / (d_j + z_k)
            xv2 = (X @ V) ** 2                          # (t, p)
            denom = d[:, None] + z[None, :]             # (p, n_z)
            h = (xv2 @ (1.0 / denom)) / t_             # (t, n_z)
        else:
            # wide case: XX'/t = U D U', H = U D(D+z)^{-1} U'
            # h_ii(z_k) = sum_j d_j/(d_j+z_k) * U[i,j]^2
            u2 = V ** 2                                  # (t, t) — V stores U here
            d_ratio = d[:, None] / (d[:, None] + z[None, :])   # (t, n_z)
            h = u2 @ d_ratio                            # (t, n_z)
        return h

    # ------------------------------------------------------------------
    # Internal solvers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_standard(
        signals: np.ndarray,
        labels: np.ndarray,
        t_: int,
        shrinkage: np.ndarray,
    ):
        """Decompose the p×p matrix S'S/t (used when p < t)."""
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels / t_          # (p,) or (p, k)
        multiplied = eigenvectors.T @ means      # (p,) or (p, k)

        denom = eigenvalues[:, None] + shrinkage[None, :]  # (p, n_z)

        if multiplied.ndim == 1:
            intermed = multiplied[:, None] / denom              # (p, n_z)
        else:
            intermed = multiplied[:, :, None] / denom[:, None, :]  # (p, k, n_z)

        betas = eigenvectors @ intermed          # (p, n_z) or (p, k, n_z)
        return betas, eigenvectors, eigenvalues

    @staticmethod
    def _fit_swap(
        signals: np.ndarray,
        labels: np.ndarray,
        t_: int,
        shrinkage: np.ndarray,
    ):
        """Decompose the t×t matrix SS'/t (used when p >= t)."""
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels / t_                      # (t,) or (t, k)
        multiplied = eigenvectors.T @ means      # (t,) or (t, k)

        denom = eigenvalues[:, None] + shrinkage[None, :]  # (t, n_z)

        if multiplied.ndim == 1:
            intermed = multiplied[:, None] / denom              # (t, n_z)
        else:
            intermed = multiplied[:, :, None] / denom[:, None, :]  # (t, k, n_z)

        XtU = signals.T @ eigenvectors           # (p, t)
        betas = XtU @ intermed                   # (p, n_z) or (p, k, n_z)
        return betas, eigenvectors, eigenvalues

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, future_signals: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Predict for all z values in shrinkage_list.

        Parameters
        ----------
        future_signals : array of shape (m, p), or None

        Returns
        -------
        predictions : array of shape (m, n_z), or None if future_signals is None
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        if future_signals is None:
            return None
        future_signals = np.asarray(future_signals, dtype=float)
        return future_signals @ self.betas   # (m, n_z) or (m, k, n_z)