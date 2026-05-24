"""Adstock transformations: geometric decay and Weibull saturation."""
import numpy as np


def apply_adstock(spend: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Geometric adstock: each week carries over a fraction of prior spend."""
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def weibull_saturation(x: np.ndarray, k: float = 2.0, lam: float = 1.0) -> np.ndarray:
    """Weibull CDF saturation: diminishing returns on high spend."""
    return 1 - np.exp(-(x / lam) ** k)