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
        row = lambda profit, income, minutes: {"profit": profit, "income": income, "online_minutes": minutes, "wage": 1.0}
        a = [row(10.0, 5.0, 100.0), row(12.0, 6.0, 100.0)]
        b = [row(13.0, 5.0, 90.0), row(15.0, 7.0, 90.0)]
        d = dec.paired_delta(b, a)
        self.assertAlmostEqual(d["profit"]["mean"], 3.0)
        self.assertAlmostEqual(d["allocation"]["mean"], -0.5)
        self.assertAlmostEqual(d["allocation"]["mean"] + d["efficiency"]["mean"], d["profit"]["mean"])
        self.assertAlmostEqual(d["supply_channel"]["mean"], 5.5 / 100.0 * 10.0)  # 10 fewer paid driver-minutes at 0.055/min

    def test_policy_families_nest_exactly(self):
        prices, shares = np.array([1.1, 1.2, 1.3, 1.4]), np.full(8, 0.7)
        for kind in [k for k in dec.KINDS if k != "f"]:
            theta = dec.embed(kind, np.full(4, 1.25), np.full(8, 0.6))
            p, s = dec.canonical(kind, theta)
            np.testing.assert_array_equal(p, np.full(4, 1.25))
            np.testing.assert_array_equal(s, np.full(8, 0.6))
        # (a') solution embeds into (b') without change
        theta_b1 = dec.embed("b1", prices, shares)
        np.testing.assert_array_equal(np.r_[dec.canonical("b1", theta_b1)[0], dec.canonical("b1", theta_b1)[1]], np.r_[prices, shares])
        self.assertEqual([dec.KINDS[k] for k in ["a", "a1", "b", "b1"]], [2, 5, 9, 12])
        # the fixed-share control (f) is nested in (a) and pins the share at 0.75
        p, s = dec.canonical("f", dec.embed("f", np.full(4, 1.3), np.full(8, 0.6)))
        np.testing.assert_array_equal(p, np.full(4, 1.3))
        np.testing.assert_array_equal(s, np.full(8, 0.75))
        self.assertEqual(dec.KINDS["f"], 1)

    def test_constraint_forms_have_unit_baseline(self):
        rows = [{"income": 100.0, "wage": 50.0, "served": 20.0, "trip_distance": 40.0, "trip_minutes": 80.0, "profit": 1.0}] * 3
        for form, value in [("hourly", 50.0), ("budget", 100.0), ("per_trip", 1.0)]:
            con = dec.Constraint(form, 1.2, rows)
            self.assertAlmostEqual(con.baseline_value, value)
            self.assertAlmostEqual(con.target, 1.2 * value)
            self.assertFalse(con.score(rows)[0] == con.score(rows)[1])  # a 20% shortfall is penalised

    def test_collapsed_day_counts_as_violation_not_nan(self):
        rows = [{"income": 100.0, "wage": 50.0, "served": 20.0, "trip_distance": 40.0, "trip_minutes": 80.0, "profit": 1.0}] * 3
        con = dec.Constraint("hourly", 1.0, rows)
        collapsed = dict(rows[0], wage=float("nan"), income=0.0)
        objective, profit, metric = con.score([rows[0], collapsed, rows[2]])
        self.assertTrue(np.isfinite(objective) and np.isfinite(metric))
        self.assertLess(objective, profit)  # penalised
        self.assertEqual(con.value(collapsed), 0.0)

    def test_decoupled_policy_maps_time_blocks_and_zone_groups(self):
        group = (SCENARIO.dd_train.sum(axis=(0, 2)) > np.median(SCENARIO.dd_train.sum(axis=(0, 2)))).astype(int)
        dec._CTX.update(base=lib.constant_policy(SCENARIO), group=group)
        prices, shares = np.array([1.0, 1.1, 1.2, 1.3]), np.arange(8) * 0.1 + 0.3
        policy = dec.make_policy(np.r_[prices, shares])
        env = lib.make_environment(SCENARIO, MARKET, 3)
        for t, block in [(0, 0), (359, 0), (360, 1), (1439, 3)]:
            action = policy.decide(env, t)
            np.testing.assert_allclose(action.share, shares.reshape(4, 2)[block][group])
            np.testing.assert_allclose(action.price, prices[block])

    def test_pay_weighted_zone_choice_follows_zone_pay(self):
        market = lib.market_namespace({}, dict(use_passenger_entry_br=True, use_driver_online_br=True, online_zone_choice="pay_weighted"))
        env = lib.make_environment(SCENARIO, market, 3)
        env.zone_profit = np.zeros(16)
        env.zone_profit[5] = 100.0
        zones = [env._choose_online_zone(u) for u in np.linspace(0.01, 0.99, 50)]
        self.assertGreater(np.mean(np.array(zones) == 5), 0.95)
        default = lib.make_environment(SCENARIO, MARKET, 3)
        self.assertEqual(default.online_zone_choice, "random")
        self.assertEqual(default._choose_online_zone(0.5), 8)

    def test_profile_concentration_shrinks_spread_only(self):
        import tempfile
        from tools.make_scenario import main as make
        with tempfile.TemporaryDirectory() as root:
            wide = np.load(make(["--name", "w", "--output_root", root, "--fleet_multiplier", "2"]) / "vp.npy")
            narrow = np.load(make(["--name", "n", "--output_root", root, "--fleet_multiplier", "2", "--profile_concentration", "0.5"]) / "vp.npy")
        self.assertAlmostEqual(narrow.mean(), wide.mean(), places=6)
        self.assertAlmostEqual(narrow.std(), wide.std() * 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
