import subprocess
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from economic_action import ActionSpec
from sim import Environment


ROOT = Path(__file__).resolve().parents[1]


def config(passenger=False, driver=False):
    return SimpleNamespace(action_mode="joint", use_passenger_entry_br=passenger,
                           use_driver_online_br=driver, seed=47, br_random_seed=47)


class JointContractTests(unittest.TestCase):
    def environment(self):
        env = Environment(np.array([[0., 2.], [2., 0.]]),
                          np.array([[1., 3.], [3., 1.]]),
                          np.array([-1., -1.]), 2, br_config=config())
        env.rand_double = [0.] * 100
        return env

    def test_action_layout_bounds_and_validation(self):
        spec = ActionSpec(2, "joint")
        action = spec.decode([-2, 2, -2, 2])
        np.testing.assert_allclose(action.price, [1 / 3, 3])
        np.testing.assert_allclose(action.compensation, [.5, 2])
        np.testing.assert_allclose(spec.decode([0] * 4).compensation, 1)
        for bounds in [(0, 2), (2, 1), (.5, np.inf)]:
            with self.assertRaises(ValueError):
                ActionSpec(2, "joint", *bounds)
        with self.assertRaises(ValueError):
            spec.decode([0, 1])

    def test_request_locks_price_and_compensation_by_passenger_origin(self):
        env = self.environment()
        demand = np.array([[0, 1], [0, 0]])
        env.step(demand, [], np.array([.5, 3]), np.array([2., .5]))
        env.step(np.zeros((2, 2), int), [(1, 0)], np.ones(2) * 3, np.ones(2) * .5)
        audit = env.get_last_step_audit()
        event = env.get_last_step_trace()["order_events"][0]
        base_fare = 2 + 2 * 2 + .4 * 3
        self.assertAlmostEqual(event["fare"], .5 * base_fare)
        self.assertAlmostEqual(event["driver_order_payment"], .75 * 2 * base_fare)
        self.assertAlmostEqual(event["pickup_cost"], .1 * 2 + .1 * 3)
        self.assertEqual(event["compensation_multiplier"], 2)
        self.assertEqual(audit["negative_margin_order_count"], 1)
        self.assertAlmostEqual(audit["negative_margin_total"], -event["order_margin"])
        self.assertAlmostEqual(audit["service_expense"], event["driver_order_payment"] + event["pickup_cost"])
        self.assertAlmostEqual(audit["driver_income_basis"], 2 * base_fare)

    def test_higher_pay_increases_existing_supply_feedback_without_changing_fare(self):
        values = []
        for q in [1., 2.]:
            env = self.environment()
            env.step(np.array([[0, 1], [0, 0]]), [], np.ones(2), np.ones(2) * q)
            outcome = env.step(np.zeros((2, 2), int), [(1, 0)], np.ones(2), np.ones(2))
            values.append((outcome[0], env.get_last_step_audit()["driver_payment_total"],
                           env.avg_profit, env.driver_online_probability(1.0)))
        self.assertEqual(values[0][0], values[1][0])
        self.assertTrue(all(values[1][i] > values[0][i] for i in [1, 2, 3]))

    def test_tied_controls_reproduce_pinned_prechange_simulator(self):
        source = subprocess.check_output(["git", "show", "2424174:sim.py"], cwd=str(ROOT), text=True)
        original = types.ModuleType("pre_joint_sim")
        exec(compile(source, "2424174:sim.py", "exec"), original.__dict__)
        for passenger, driver in [(False, False), (True, False), (False, True), (True, True)]:
            trajectories = []
            for cls in [original.Environment, Environment]:
                np.random.seed(47)
                env = cls(np.array([[0., 2.], [2., 0.]]), np.ones((2, 2)) * 2,
                          np.array([-.2, .3, 1.1, 2.]), 2, br_config=config(passenger, driver))
                rows = []
                for t in range(120):
                    if t == 60:
                        env.reset()
                    p = np.array([1 / 3, 3.]) if t % 2 else np.array([2., .7])
                    schedule = []
                    for zone in range(2):
                        schedule.extend([(zone, zone)] * min(int(env.veh_count[zone]),
                            int(env.pass_count[:, zone, :].sum())))
                    demand = env.apply_passenger_bounded_rationality(np.ones((2, 2), int), p)
                    outcome = env.step(demand, schedule, p, p) if cls is Environment else env.step(demand, schedule, p)
                    rows.append((outcome, env.avg_profit, env.zone_profit.copy(), env.pass_count.copy(),
                                 [(v.state, v.loc, v.remaining_time, v.waiting) for v in env.veh_list]))
                trajectories.append(rows)
            for old, new in zip(*trajectories):
                for index in range(4):
                    np.testing.assert_allclose(old[index], new[index], rtol=1e-12, atol=1e-12)
                self.assertEqual(old[4], new[4])

    def test_invalid_control_is_rejected_before_state_changes(self):
        env = self.environment()
        before = env.pass_count.copy()
        for q in [[0, 1], [1], [np.nan, 1]]:
            with self.assertRaises(ValueError):
                env.step(np.ones((2, 2), int), [], np.ones(2), q)
            np.testing.assert_array_equal(env.pass_count, before)


if __name__ == "__main__":
    unittest.main()
