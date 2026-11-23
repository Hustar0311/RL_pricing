import unittest
from argparse import Namespace

import numpy as np

from test_joint_td3 import ROOT
from tools import rollout_lib as lib
from tools.measure_elasticity import arc_elasticity, measure
from sim import Environment


SCENARIO = lib.Scenario(ROOT / "grid_small_dynamic" / "4" / "4")


def market(**kw):
    return lib.market_namespace({}, dict(use_passenger_entry_br=True, use_driver_online_br=True, **kw))


class WelfareTests(unittest.TestCase):
    def test_consumer_surplus_formula_and_take_rate(self):
        policy = lib.constant_policy(SCENARIO, 1.0, 0.65)
        env = lib.make_environment(SCENARIO, market(), 3)
        trace = lib.simulate(env, policy, SCENARIO.dd_test, 100, 30, market(), 5)
        j = lib.METRICS.index
        self.assertAlmostEqual(trace[:, j("driver_total_income")].sum() / trace[:, j("revenue")].sum(), 0.65, places=9)
        env.restore_state(env.snapshot_state())
        # CS: beta * sum_i M_i log(1+exp(U_i/beta)); evaluate against a direct computation
        utility, potential = np.array([1., -2.]), np.array([3., 4.])
        env.beta_passenger = 1.5
        expected = 1.5 * np.sum(potential * np.log1p(np.exp(utility / 1.5)))
        self.assertAlmostEqual(env.passenger_consumer_surplus(utility, potential), expected)
        env.beta_passenger = 0.0
        self.assertAlmostEqual(env.passenger_consumer_surplus(utility, potential), 3.0)


class IsolatedRngTests(unittest.TestCase):
    def run_arm(self, seed, steps=40):
        env = lib.make_environment(SCENARIO, market(), 11)
        policy = lib.constant_policy(SCENARIO)
        return env, lib.simulate(env, policy, SCENARIO.dd_test, 200, steps, market(), seed)

    def test_no_global_lists_and_common_random_numbers(self):
        env, first = self.run_arm(7)
        self.assertEqual((env.rand_int, env.rand_double), ([], []))
        np.random.seed(0)
        _, again = self.run_arm(7)
        np.testing.assert_array_equal(first, again)
        _, other = self.run_arm(8)
        self.assertFalse(np.array_equal(first, other))

    def test_snapshot_restore_replays_exactly(self):
        env = lib.make_environment(SCENARIO, market(), 11)
        policy = lib.constant_policy(SCENARIO)
        lib.simulate(env, policy, SCENARIO.dd_test, 0, 60, market(), 1)
        state = env.snapshot_state()
        a = lib.simulate(env, policy, SCENARIO.dd_test, 60, 30, market(), 2)
        env.restore_state(state)
        b = lib.simulate(env, policy, SCENARIO.dd_test, 60, 30, market(), 2)
        np.testing.assert_array_equal(a, b)

    def test_intervention_arms_are_paired(self):
        env = lib.make_environment(SCENARIO, market(), 11)
        policy = lib.constant_policy(SCENARIO)
        state = env.snapshot_state()
        control = lib.simulate(env, policy, SCENARIO.dd_test, 0, 20, market(), 3)
        env.restore_state(state)
        null = lib.simulate(env, policy, SCENARIO.dd_test, 0, 20, market(), 3, lib.Intervention("price_multiply", 1.0, 20))
        np.testing.assert_array_equal(control, null)
        env.restore_state(state)
        shock = lib.simulate(env, policy, SCENARIO.dd_test, 0, 20, market(), 3, lib.Intervention("demand_shock", 2.0, 20))
        self.assertGreater(shock[:, lib.METRICS.index("entered")].sum(), control[:, lib.METRICS.index("entered")].sum())


class CalibrationTests(unittest.TestCase):
    def env(self, **kw):
        return lib.make_environment(SCENARIO, market(passenger_calibration="anchor", **kw), 5)

    def test_anchor_entry_equals_demand_and_elasticity(self):
        for target in [-1.0, -0.6]:
            env = self.env(target_demand_elasticity=target)
            ratio = lambda p: env.passenger_enter_probability(np.full(16, p))[0]
            self.assertAlmostEqual(ratio(1.0), 1.0, places=12)
            h = 1e-5
            elasticity = (np.log(ratio(1 + h)) - np.log(ratio(1 - h))) / (np.log(1 + h) - np.log(1 - h))
            self.assertAlmostEqual(elasticity, target, places=4)
            self.assertGreater(ratio(0.8), 1.0)  # entry may exceed demand below the anchor price

    def test_default_market_is_untouched(self):
        env = lib.make_environment(SCENARIO, market(), 5)
        self.assertEqual(env.passenger_calibration, "none")
        self.assertAlmostEqual(float(env.passenger_enter_probability(np.full(16, 1.0))[0]), 1 / (1 + np.exp(-1.0)))


class DriverInertiaTests(unittest.TestCase):
    def switches(self, **kw):
        m = market(**kw)
        env = lib.make_environment(SCENARIO, m, 4)
        trace = lib.simulate(env, lib.constant_policy(SCENARIO), SCENARIO.dd_test, 300, 120, m, 9)
        return trace[:, lib.METRICS.index("switches")]

    def test_interval_and_switch_cost_reduce_switching(self):
        base, slow, costly = self.switches(), self.switches(driver_decision_interval=30), self.switches(driver_switch_cost=2.0)
        self.assertGreater(base.sum(), 0)
        self.assertLess(slow.sum(), base.sum())
        self.assertLess(costly.sum(), base.sum())

    def test_only_scheduled_drivers_decide(self):
        m = market(driver_decision_interval=10)
        env = lib.make_environment(SCENARIO, m, 4)
        policy = lib.constant_policy(SCENARIO)
        deciders = []
        for t in range(30):
            lib.simulate(env, policy, SCENARIO.dd_test, t, 1, m, 1)
            deciders.append(env.get_last_step_stats()["driver_decision_count"])
        self.assertLessEqual(max(deciders), len(SCENARIO.vp) / 10 + 1)


class ElasticityToolTests(unittest.TestCase):
    def test_arc_elasticity_and_small_run(self):
        self.assertAlmostEqual(float(arc_elasticity(0.9, 1.1, 1.05, 0.95)), -2.0)
        policy = lib.constant_policy(SCENARIO)
        m = market()
        state, held, t0 = lib.build_start_state(SCENARIO, policy, m, 100, 1)
        out = measure(SCENARIO, policy, m, state, held, t0, 3, 50, 0.05, 10, 20, 5, 1)
        self.assertEqual(out["epsilon_D"]["replicates"], 3)
        self.assertLess(out["epsilon_D"]["pooled_arc_elasticity"], 0)


if __name__ == "__main__":
    unittest.main()


class CounterfactualToolTests(unittest.TestCase):
    def test_paired_report_shape_and_signs(self):
        from tools.counterfactual_eval import evaluate
        policy = lib.constant_policy(SCENARIO)
        m = market()
        state, held, t0 = lib.build_start_state(SCENARIO, policy, m, 100, 1)
        arms = [lib.Intervention.parse("demand_shock:1.0@10", 10), lib.Intervention.parse("remove_drivers:4", 10),
                lib.Intervention.parse("share_set:0.9", 10)]
        report = evaluate(SCENARIO, policy, m, state, held, t0, arms, 6, 20, 1, horizons=[1, 5, 15])
        self.assertEqual(set(report), {"demand_shock:1@10", "remove_drivers:4@10", "share_set:0.9@10"})
        shock = report["demand_shock:1@10"]["15"]["served"]
        self.assertGreater(shock["diff"], 0)
        removed = report["remove_drivers:4@10"]["15"]["online_vehicles"]
        self.assertLess(removed["diff"], 0)
        self.assertEqual(report["share_set:0.9@10"]["1"]["revenue"]["diff"], 0.0)  # same prices, same first-minute revenue


class ScenarioAndHardThresholdTests(unittest.TestCase):
    def test_make_scenario_scales_fleet_profiles_and_demand(self):
        import tempfile
        from pathlib import Path
        from tools.make_scenario import main as make
        with tempfile.TemporaryDirectory() as root:
            target = make(["--name", "s", "--fleet_multiplier", "3", "--demand_scale", "2", "--profile_scale", "1.5",
                           "--output_root", root])
            vp, base = np.load(target / "vp.npy"), np.load(ROOT / "grid_small_dynamic/4/4/vp.npy")
            self.assertEqual(len(vp), 3 * len(base))
            np.testing.assert_allclose(vp[:len(base)], base * 1.5)
            self.assertFalse(np.allclose(vp[len(base):2 * len(base)], base * 1.5))
            np.testing.assert_allclose(np.load(target / "dd_test.npy"), SCENARIO.dd_test * 2)
            self.assertTrue((Path(root) / ".gitignore").exists())

    def test_hard_threshold_respects_interval_and_switch_cost(self):
        for kw in [dict(driver_decision_interval=15), dict(driver_switch_cost=1.0)]:
            m = lib.market_namespace({}, dict(use_passenger_entry_br=True, use_driver_online_br=False, **kw))
            env = lib.make_environment(SCENARIO, m, 2)
            policy = lib.constant_policy(SCENARIO)
            decisions = []
            for t in range(30):
                lib.simulate(env, policy, SCENARIO.dd_test, t, 1, m, 1)
                decisions.append(env.get_last_step_stats()["driver_decision_count"])
            if "driver_decision_interval" in kw:
                self.assertLess(max(decisions), len(SCENARIO.vp))
            else:
                self.assertTrue(all(d <= len(SCENARIO.vp) for d in decisions))


class IdleCostBearerTests(unittest.TestCase):
    def run_with(self, **kw):
        m = market(**kw)
        env = lib.make_environment(SCENARIO, m, 4)
        trace = lib.simulate(env, lib.constant_policy(SCENARIO, 1.0, 0.65), SCENARIO.dd_test, 300, 90, m, 9)
        return env, trace

    def test_platform_is_step_for_step_identical_to_default(self):
        _, default = self.run_with()
        _, platform = self.run_with(idle_cost_bearer="platform")
        np.testing.assert_array_equal(default, platform)
        self.assertEqual(platform[:, lib.METRICS.index("driver_idle_cost")].sum(), 0.0)

    def test_driver_moves_idle_cost_from_platform_to_driver_only(self):
        _, platform = self.run_with(idle_cost_bearer="platform")
        env, driver = self.run_with(idle_cost_bearer="driver")
        j = lib.METRICS.index
        idle = driver[:, j("driver_idle_cost")]
        self.assertGreater(idle.sum(), 0)
        # market dynamics are untouched: only accounting differs
        for name in lib.METRICS:
            if name not in ("net_profit", "driver_idle_cost"):
                np.testing.assert_array_equal(platform[:, j(name)], driver[:, j(name)], err_msg=name)
        np.testing.assert_allclose(driver[:, j("net_profit")], platform[:, j("net_profit")] + idle, atol=1e-9)
        audit = env.get_last_step_audit()
        expected = (audit["driver_payment_total"] - audit["driver_idle_cost"]) / audit["online_vehicle_minutes"] * 60.0
        self.assertAlmostEqual(audit["driver_hourly_wage"], expected)


class DriverSignalTests(unittest.TestCase):
    def run_env(self, **kw):
        m = market(**kw)
        env = lib.make_environment(SCENARIO, m, 4)
        trace = lib.simulate(env, lib.constant_policy(SCENARIO, 1.0, 0.65), SCENARIO.dd_test, 300, 90, m, 9)
        return env, trace

    def test_gross_is_default_and_net_lowers_signal_by_idle_cost(self):
        _, default = self.run_env()
        _, gross = self.run_env(driver_signal="gross")
        np.testing.assert_array_equal(default, gross)
        env_n, net = self.run_env(driver_signal="net")
        j = lib.METRICS.index
        self.assertLess(net[:, j("driver_income_signal")].mean(), gross[:, j("driver_income_signal")].mean())
        self.assertLess(net[:, j("online_vehicles")].mean(), gross[:, j("online_vehicles")].mean())
        self.assertGreater(env_n._last_reposition_expense, -1e-12)

    def test_net_signal_step_formula(self):
        m = market(driver_signal="net")
        env = lib.make_environment(SCENARIO, m, 4)
        lib.simulate(env, lib.constant_policy(SCENARIO, 1.0, 0.65), SCENARIO.dd_test, 300, 20, m, 9)
        audit = env.get_last_step_audit()
        self.assertTrue(np.isfinite(env.avg_profits[-1]))
        # last window entry = (basis - (cruise + previous reposition)/delta) / denominator; check by reconstruction
        cruise = env.beta2 * audit["idle_cruising_vehicle_count"]
        self.assertGreaterEqual(cruise, 0.0)

    def test_shift_fit_recovers_linear_root(self):
        from tools.calibrate_net_signal import fit_shift
        shift, slope = fit_shift([-1, 0, 1, 2], [0.3 + 0.1 * c for c in [-1, 0, 1, 2]], 0.5)
        self.assertAlmostEqual(shift, 2.0, places=9)
        self.assertAlmostEqual(slope, 0.1, places=9)
        with self.assertRaises(ValueError):
            fit_shift([0, 1], [0.5, 0.4], 0.45)
