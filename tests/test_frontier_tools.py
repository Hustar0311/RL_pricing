import unittest

import numpy as np

from tools import frontier_decoupling as fr


class FrontierArithmeticTests(unittest.TestCase):
    def test_interpolation_and_range(self):
        g, p = np.array([1.2, 0.8, 1.0]), np.array([10.0, 30.0, 20.0])
        out = fr.interpolate(g, p, [0.8, 0.9, 1.1, 1.3])
        np.testing.assert_allclose(out[:3], [30.0, 25.0, 15.0])
        self.assertTrue(np.isnan(out[3]))

    def test_delta_and_bootstrap_of_a_constant_gap(self):
        rng = np.random.default_rng(0)
        g = np.tile(np.array([0.8, 1.0, 1.25])[:, None], (1, 12))
        p_a = np.array([30.0, 20.0, 10.0])[:, None] + rng.normal(0, 0.1, (3, 12))
        p_b = p_a + 5.0
        point, lo, hi = fr.bootstrap_delta(g, p_a, g, p_b, [0.8, 1.0, 1.2], n_boot=200)
        np.testing.assert_allclose(point, 5.0, atol=1e-9)
        self.assertTrue(np.all(lo <= point + 1e-9) and np.all(hi >= point - 1e-9))

    def test_lambda_grid_hits_targets(self):
        pilot = [(0.0, 0.5), (1e5, 0.9), (2e5, 1.3)]
        grid = fr.choose_lambdas(pilot)
        self.assertEqual(grid, sorted(grid))
        self.assertAlmostEqual(np.interp(1.0, [0.5, 0.9, 1.3], [0.0, 1e5, 2e5]), 1.25e5)
        self.assertIn(1.25e5, grid)

    def test_fill_lambdas_extends_grid_where_a_frontier_misses_targets(self):
        def point(lam, ga, gb):
            return {"lambda": lam, "a": {"days": {"g_rel": [ga] * 3}}, "b": {"days": {"g_rel": [gb] * 3}}}
        short = {"points": [point(1000.0, 0.9, 0.7), point(2000.0, 1.0, 0.75)]}
        new = fr.fill_lambdas(short)
        self.assertTrue(new and max(new) > 2000.0)  # (b) never reaches 1.2 -> push lambda up
        covered = {"points": [point(1000.0, 0.7, 0.7), point(2000.0, 0.85, 0.85), point(3000.0, 1.0, 1.0), point(4000.0, 1.15, 1.15), point(5000.0, 1.3, 1.3)]}
        self.assertEqual(fr.fill_lambdas(covered), [])
        jumpy = {"points": [point(1000.0, 0.7, 0.7), point(2000.0, 1.3, 1.3)]}
        self.assertEqual(fr.fill_lambdas(jumpy), [1500.0])


if __name__ == "__main__":
    unittest.main()
