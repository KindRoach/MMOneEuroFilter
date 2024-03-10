# Modified from https://github.com/jaantollander/OneEuroFilter/blob/master/python/one_euro_filter.py

import numpy as np


class OneEuroFilterMD:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """Initialize the one euro filter."""

        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values.
        self.has_prev = False
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if not self.has_prev:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = np.zeros_like(x)
            self.has_prev = True
            return x

        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    @staticmethod
    def smoothing_factor(t_e: float, cutoff: np.ndarray):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        return a * x + (1 - a) * x_prev
