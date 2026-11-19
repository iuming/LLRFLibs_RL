###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Sample-efficient scalar-parameter tuning with a Bayesian optimizer.

A dependency-free Gaussian-process (RBF kernel) with the upper-confidence-bound
acquisition, mirroring the role that toolkits like ``Xopt`` (LBNL) play on real
accelerators: instead of RL, a classical optimizer searches low-dimensional
setpoints (tuner offsets, PI gains, gain-scheduling setpoints) sample
efficiently.  The interface is deliberately small so both black-box objective
functions and environment rewards can be plugged in.

Typical use::

    best_kp, history = tune_scalar(
        objective=lambda kp: -evaluate_pi_policy(kp),  # maximize reward
        bounds=(0.01, 0.5), n_init=3, n_iters=10, seed=0,
    )
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["GaussianProcess", "BayesianOptimizer", "tune_scalar"]


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length: float, signal_var: float) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    dist2 = (x1[:, None] - x2[None, :]) ** 2 / (2.0 * length**2)
    return signal_var * np.exp(-dist2)


class GaussianProcess:
    """Gaussian-process regression with an RBF kernel (numpy only).

    Args:
        length:      RBF length scale.
        signal_var:  Signal variance.
        noise_var:   Observed-noise variance (jitter stabilizes the solve).
    """

    def __init__(self, length: float = 0.3, signal_var: float = 1.0, noise_var: float = 1e-3) -> None:
        self.length = float(length)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._k: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the posterior on ``(x, y)`` pairs.

        Args:
            x: 1-D input points.
            y: Scalar outputs.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        k = _rbf_kernel(x, x, self.length, self.signal_var) + self.noise_var * np.eye(x.shape[0])
        # Least-squares solve never raises, even for near-singular kernels.
        self._k = k
        self._alpha = np.linalg.lstsq(k, y, rcond=None)[0]
        self._x = x
        self._y = y

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at new points.

        Args:
            x: 1-D query points.

        Returns:
            ``(mean, std)`` arrays.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        if self._x is None or self._y is None or self._k is None:
            return np.zeros(x.shape[0]), np.full(x.shape[0], np.inf)
        k_star = _rbf_kernel(x, self._x, self.length, self.signal_var)
        mean = k_star @ self._alpha
        # Posterior variance via the Cholesky factor of the (regularized) fit.
        chol = np.linalg.cholesky(self._k)
        v = np.linalg.lstsq(chol, k_star.T, rcond=None)[0]
        var = self.signal_var - np.sum(v**2, axis=0)
        return mean.ravel(), np.sqrt(np.clip(var, 0.0, None))


class BayesianOptimizer:
    """GP-based Bayesian optimizer with the UCB acquisition.

    Args:
        bounds:      ``(low, high)`` of the scalar search space.
        n_init:      Number of initial random evaluations before the GP fit.
        beta:        UCB exploration weight.
        length:      GP RBF length scale.
        signal_var:  GP signal variance.
        noise_var:   GP noise variance.
        seed:        Random seed for the initial design.
    """

    def __init__(
        self,
        bounds: tuple[float, float],
        n_init: int = 3,
        beta: float = 1.5,
        length: float = 0.2,
        signal_var: float = 1.0,
        noise_var: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.n_init = int(n_init)
        self.beta = float(beta)
        self.gp = GaussianProcess(length, signal_var, noise_var)
        self._rng = np.random.default_rng(seed)
        self._x: list[float] = []
        self._y: list[float] = []
        self._phase = "init"

    # ------------------------------------------------------------------

    def suggest(self) -> float:
        """Return the next point to evaluate.

        During the initial phase a random point is returned; afterwards the
        UCB maximum over a dense grid.

        Returns:
            A scalar within ``bounds``.
        """
        if len(self._x) < self.n_init:
            return float(self._rng.uniform(self.bounds[0], self.bounds[1]))
        grid = np.linspace(self.bounds[0], self.bounds[1], 201)
        mean, std = self.gp.predict(grid)
        ucb = mean + self.beta * std
        return float(grid[int(np.argmax(ucb))])

    def update(self, x: float, y: float) -> None:
        """Record an evaluation and refit the surrogate.

        Args:
            x: Evaluated point.
            y: Its (maximized) objective value.
        """
        self._x.append(float(x))
        self._y.append(float(y))
        if len(self._x) >= self.n_init:
            self.gp.fit(np.asarray(self._x), np.asarray(self._y))

    def best(self) -> tuple[float, float]:
        """Best evaluated point so far.

        Returns:
            ``(x_best, y_best)``.
        """
        idx = int(np.argmax(np.asarray(self._y)))
        return self._x[idx], self._y[idx]


def tune_scalar(
    objective: Callable[[float], float],
    bounds: tuple[float, float],
    n_init: int = 3,
    n_iters: int = 10,
    seed: int = 0,
) -> tuple[float, list[float]]:
    """Maximize a scalar objective with the Bayesian optimizer.

    Args:
        objective: Black-box function to maximize (noisy evaluations allowed).
        bounds:    ``(low, high)`` search range.
        n_init:    Initial random evaluations.
        n_iters:   Total evaluations (including the initial ones).
        seed:      Random seed.

    Returns:
        ``(best_x, history_y)``.
    """
    opt = BayesianOptimizer(bounds, n_init=n_init, seed=seed)
    history: list[float] = []
    for _ in range(int(n_iters)):
        x = opt.suggest()
        y = float(objective(x))
        opt.update(x, y)
        history.append(y)
    return opt.best()[0], history
