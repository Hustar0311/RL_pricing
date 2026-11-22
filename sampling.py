"""Count samplers shared by the runners and the offline evaluation tools.

Every function takes an optional ``numpy.random.Generator``.  ``None`` keeps the legacy
behaviour of drawing from the global ``numpy.random`` state, so default runs stay identical.
"""
import numpy as np


def sample_fractional_counts(expected_counts, rng=None):
    expected_counts = np.maximum(np.asarray(expected_counts, dtype=float), 0.0)
    base_counts = np.floor(expected_counts).astype(int)
    residual = np.clip(expected_counts - base_counts, 0.0, 1.0)
    draw = np.random.binomial if rng is None else rng.binomial
    return base_counts + draw(1, residual)


def sample_poisson_counts(demand_rates, rng=None):
    demand_rates = np.asarray(demand_rates, dtype=float)
    sampled = np.zeros(demand_rates.shape, dtype=int)
    ind = demand_rates > 0
    draw = np.random.poisson if rng is None else rng.poisson
    sampled[ind] = draw(demand_rates[ind])
    return sampled
