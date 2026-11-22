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
