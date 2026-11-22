import unittest

import numpy as np

from test_joint_td3 import ROOT
from tools import rollout_lib as lib
from tools import static_decoupling as dec
from tools.static_sweep import optimum

SCENARIO = lib.Scenario(ROOT / "grid_small_dynamic" / "4" / "4")
MARKET = lib.market_namespace({}, dict(use_passenger_entry_br=True, use_driver_online_br=True))


class StaticToolTests(unittest.TestCase):
    def test_service_quality_metrics_are_consistent(self):
        env = lib.make_environment(SCENARIO, MARKET, 3)
        trace = lib.simulate(env, lib.constant_policy(SCENARIO), SCENARIO.dd_test, 200, 120, MARKET, 5)
        col = lib.METRICS.index
        served, entered = trace[:, col("served")].sum(), trace[:, col("entered")].sum()
        self.assertGreater(served, 0)
        self.assertLessEqual(served, entered + 50)  # carried-over backlog may be served
        self.assertGreaterEqual(trace[:, col("wait_minutes")].sum(), 0)
        self.assertGreater(trace[:, col("pickup_minutes")].sum(), 0)
        self.assertGreaterEqual(trace[:, col("left")].sum(), 0)

    def test_time_varying_share_policy(self):
        calls = []
        policy = lib.constant_policy(SCENARIO)
        policy.constant = lambda t: (1.0, np.where(np.arange(16) < 8, 0.5, 0.9)) if t < 60 else (1.0, 0.75)
        env = lib.make_environment(SCENARIO, MARKET, 3)
        action = policy.decide(env, 10)
        np.testing.assert_allclose(action.share[:8], 0.5)
        np.testing.assert_allclose(action.compensation[8:], 0.9 / 0.75)
        late = policy.decide(env, 100)
        np.testing.assert_allclose(late.compensation, 1.0)
        trace = lib.simulate(env, policy, SCENARIO.dd_test, 0, 90, MARKET, 5)
        j = lib.METRICS.index
        # take rate differs between the first block (mixed shares) and the second (0.75 everywhere)
        early = 1 - trace[:60, j("driver_total_income")].sum() / trace[:60, j("revenue")].sum()
        later = 1 - trace[60:, j("driver_total_income")].sum() / trace[60:, j("revenue")].sum()
        self.assertAlmostEqual(later, 0.25, places=9)
        self.assertNotAlmostEqual(early, 0.25, places=3)

    def test_wage_floor_filters_optimum(self):
        grid = {"1|0.5": {"profit": {"mean": 10.0}, "hourly_wage": {"mean": 40.0}},
                "1|0.7": {"profit": {"mean": 8.0}, "hourly_wage": {"mean": 60.0}},
                "1|0.9": {"profit": {"mean": 2.0}, "hourly_wage": {"mean": 80.0}}}
        self.assertEqual(optimum(grid, [1.0], [0.5, 0.7, 0.9])[:2], (1.0, 0.5))
        p, s, interior, profit = optimum(grid, [1.0], [0.5, 0.7, 0.9], wage_floor=55.0)
        self.assertEqual((s, profit), (0.7, 8.0))

    def test_paired_delta_splits_into_allocation_and_efficiency(self):
        a = [{"profit": 10.0, "income": 5.0, "revenue": 20.0}, {"profit": 12.0, "income": 6.0, "revenue": 22.0}]
        b = [{"profit": 13.0, "income": 5.0, "revenue": 23.0}, {"profit": 15.0, "income": 7.0, "revenue": 26.0}]
        d = dec.paired_delta(b, a)
        self.assertAlmostEqual(d["profit"]["mean"], 3.0)
        self.assertAlmostEqual(d["allocation"]["mean"], -0.5)
        self.assertAlmostEqual(d["allocation"]["mean"] + d["efficiency"]["mean"], d["profit"]["mean"])

    def test_decoupled_policy_maps_time_blocks_and_zone_groups(self):
        group = (SCENARIO.dd_train.sum(axis=(0, 2)) > np.median(SCENARIO.dd_train.sum(axis=(0, 2)))).astype(int)
        dec._CTX.update(base=lib.constant_policy(SCENARIO), group=group)
        theta = [1.3] + [0.1 * (i + 1) + 0.3 for i in range(8)]  # shares 0.4 ... 1.1 (unclipped test values)
        policy = dec.make_policy("decoupled", theta)
        env = lib.make_environment(SCENARIO, MARKET, 3)
        for t, block in [(0, 0), (359, 0), (360, 1), (1439, 3)]:
            action = policy.decide(env, t)
            expected = np.array(theta[1:]).reshape(4, 2)[block][group]
            np.testing.assert_allclose(action.share, expected)
            np.testing.assert_allclose(action.price, 1.3)


if __name__ == "__main__":
    unittest.main()
