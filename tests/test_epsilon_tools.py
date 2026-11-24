import unittest

import numpy as np

from tools import epsilon_decoupling as eps

ROWS = [{"income": 100.0, "wage": 50.0, "served": 20.0, "trip_distance": 40.0, "trip_minutes": 80.0, "profit": 1000.0,
         "online_minutes": 500.0}] * 3


class EpsilonConstraintTests(unittest.TestCase):
    def row(self, wage, om=500.0):
        return dict(ROWS[0], wage=wage, online_minutes=om)

    def test_penalty_only_outside_band(self):
        con = eps.Epsilon("hourly", ROWS, 1.2)
        inside = con.score([self.row(50.0 * 1.21)] * 3)
        self.assertAlmostEqual(inside[0], inside[1])
        below, above = con.score([self.row(50.0 * 1.1)] * 3), con.score([self.row(50.0 * 1.3)] * 3)
        self.assertLess(below[0], below[1] - 1.0)
        self.assertLess(above[0], above[1] - 1.0)
        self.assertAlmostEqual(below[1] - below[0], eps.MU * 0.1 ** 2, delta=1.0)

    def test_online_floor_penalises_fewer_minutes_only(self):
        con = eps.Epsilon("hourly", ROWS, 1.0, om_floor=500.0)
        self.assertAlmostEqual(con.score([self.row(51.0, 520.0)] * 3)[0], 1000.0)
        self.assertLess(con.score([self.row(51.0, 450.0)] * 3)[0], 1000.0 - 1.0)

    def test_boot_mean_interval_brackets_mean(self):
        out = eps.boot_mean(np.arange(20.0), 300)
        self.assertLessEqual(out["lo"], out["mean"])
        self.assertGreaterEqual(out["hi"], out["mean"])


if __name__ == "__main__":
    unittest.main()
