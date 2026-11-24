import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from center import Platform, observation_normalization
from joint_checkpoint import save_checkpoint
from main import get_arguments
from sim import Environment
from tools import rollout_lib as lib
from tools.counterfactual_eval import evaluate, format_table

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "grid_small_dynamic" / "4" / "4"
SCENARIO = lib.Scenario(FOLDER)
torch.set_num_threads(1)


def market(**kw):
    return lib.market_namespace({}, dict(dict(use_passenger_entry_br=True, use_driver_online_br=True), **kw))


def training_controller(args):
    """The controller exactly as main.py's joint branch builds it (statistics spelled out, not via the helper)."""
    s = SCENARIO
    with contextlib.redirect_stdout(io.StringIO()):
        controller = Platform(s.td, s.tt, s.num_zone, option=args.pricing_alg, actor_lr=args.actor_lr,
            critic_lr=args.critic_lr, device="cpu", update_freq=args.frequency, veh_num=len(s.vp),
            demand_mean=np.mean(s.dd_train, axis=0), demand_std=np.std(s.dd_train, axis=0),
            wage_mean=np.mean(s.vp), wage_std=np.std(s.vp), permutation=s.permutation, permutation2=s.permutation2,
            od_permutation=args.od_permutation, kernel_size=args.kernel_size, stride=args.stride, pooling=args.pooling,
            searching=args.searching, policy_delay=args.policy_delay, position_encode=args.position_encode,
            auto_correlation=args.auto_correlation, forget=args.forget, T_train=1440, action_mode=args.action_mode,
            compensation_min=args.compensation_min, compensation_max=args.compensation_max, share_min=args.share_min,
            share_max=args.share_max, share_value=args.share_value, share_lr=args.share_lr, share_init=args.share_init,
            share_head_separate=args.share_head_separate)
        controller.pricer.initialize(torch.Generator().manual_seed(3))
    return controller


class LoadPolicyTests(unittest.TestCase):
    def check(self, extra):
        args = get_arguments(["--action_mode", "joint_share", "--pricing_alg", "TD3_MLP", "--position_encode", "8",
                              "--policy_delay", "1", "--data_folder", str(FOLDER)] + extra)
        controller = training_controller(args)
        env = lib.make_environment(SCENARIO, market(), 11)
        warm = lib.constant_policy(SCENARIO, 1.2, 0.7)
        lib.simulate(env, warm, SCENARIO.dd_test, 0, 60, market(), 1)  # a non-trivial observation
        with tempfile.TemporaryDirectory() as folder:
            args.store_model_folder = folder
            save_checkpoint(controller, env, args, 1)
            policy, _ = lib.load_policy(folder, 1, SCENARIO)
        with torch.no_grad():
            expected = controller.update_action(env, 61, "test", args.od_permutation)
            loaded = policy.decide(env, 61)
        for field in ("price", "compensation", "share"):
            np.testing.assert_allclose(getattr(loaded, field), getattr(expected, field), atol=1e-6, rtol=0)
        return controller, policy, env

    def test_normalisation_matches_training(self):
        controller, policy, env = self.check([])
        stats = observation_normalization(SCENARIO.dd_train, SCENARIO.vp)
        np.testing.assert_array_equal(policy.controller.demand_mean, stats["demand_mean"])
        np.testing.assert_array_equal(policy.controller.demand_std, controller.demand_std)
        self.assertEqual(np.ndim(policy.controller.wage_mean), 0)
        self.assertEqual(policy.controller.wage_std, np.std(SCENARIO.vp))
        self.assertGreater(np.abs(policy.controller.demand_mean).max(), 0)  # default 0/1 would be a silent miss

    def test_separate_share_head_is_rebuilt(self):
        _, policy, _ = self.check(["--share_head_separate", "--share_lr", "0.002", "--share_init", "0.6"])
        self.assertEqual(policy.controller.pricer.actor.separate_head_dim, SCENARIO.num_zone)
        self.assertTrue(hasattr(policy.controller.pricer.actor, "share_head"))

    def test_data_folder_mismatch_warns(self):
        args = get_arguments(["--action_mode", "joint", "--pricing_alg", "TD3_MLP", "--position_encode", "8",
                              "--data_folder", "somewhere/else"])
        controller = training_controller(args)
        with tempfile.TemporaryDirectory() as folder:
            args.store_model_folder = folder
            save_checkpoint(controller, lib.make_environment(SCENARIO, market(), 1), args, 1)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                lib.load_policy(folder, 1, SCENARIO)
        self.assertIn("observation standardisation", stderr.getvalue())


class SnapshotMarketTests(unittest.TestCase):
    def snapshot(self, **kw):
        env = lib.make_environment(SCENARIO, market(**kw), 11)
        lib.simulate(env, lib.constant_policy(SCENARIO), SCENARIO.dd_test, 0, 20, market(**kw), 1)
        return env.snapshot_state()

    def test_matching_snapshot_passes_untouched(self):
        state = self.snapshot()
        self.assertIs(lib.align_snapshot_market(SCENARIO, state, market(), "error"), state)

    def test_mismatch_errors_and_lists_fields(self):
        state = self.snapshot(passenger_calibration="anchor", beta_passenger=0.5, driver_decision_interval=3)
        with self.assertRaises(lib.SnapshotMarketMismatch) as caught:
            lib.align_snapshot_market(SCENARIO, state, market(), "error")
        text = str(caught.exception)
        for name in ("passenger_calibration", "beta_passenger", "driver_decision_interval"):
            self.assertIn(name, text)
        self.assertNotIn("idle_cost_bearer", text)

    def test_override_rewrites_rules_and_keeps_dynamic_state(self):
        state = self.snapshot(passenger_calibration="anchor", driver_switch_cost=0.3, idle_cost_bearer="driver")
        target = market(use_driver_online_br=False)
        fixed = lib.align_snapshot_market(SCENARIO, state, target, "override")
        env = lib.make_environment(SCENARIO, target, 0)
        env.restore_state(fixed)
        self.assertEqual(env.market_rule_values(), lib.make_environment(SCENARIO, target, 0).market_rule_values())
        self.assertFalse(env.use_driver_online_br)
        self.assertNotIn("calibrated_price_sensitivity", env.__dict__)
        self.assertEqual(env.step_index, state["step_index"])
        np.testing.assert_array_equal(env.veh_count, state["veh_count"])
        self.assertEqual(lib.snapshot_market_differences(SCENARIO, fixed, target), {})

    def test_override_recomputes_anchor_calibration(self):
        state = self.snapshot()
        target = market(passenger_calibration="anchor", target_demand_elasticity=-0.6)
        env = lib.make_environment(SCENARIO, target, 0)
        env.restore_state(lib.align_snapshot_market(SCENARIO, state, target, "override"))
        self.assertAlmostEqual(env.calibrated_price_sensitivity,
                               lib.make_environment(SCENARIO, target, 0).calibrated_price_sensitivity)


class RecordActionsTests(unittest.TestCase):
    def test_default_return_is_unchanged_and_actions_have_expected_content(self):
        env = lib.make_environment(SCENARIO, market(), 11)
        state = env.snapshot_state()
        policy = lib.constant_policy(SCENARIO, 1.0, 0.6)
        plain = lib.simulate(env, policy, SCENARIO.dd_test, 0, 12, market(), 4)
        env.restore_state(state)
        arm = lib.Intervention("price_multiply", 1.1, 5)
        trace, actions = lib.simulate(env, policy, SCENARIO.dd_test, 0, 12, market(), 4, arm, None, record_actions=True)
        env.restore_state(state)
        np.testing.assert_array_equal(plain, lib.simulate(env, policy, SCENARIO.dd_test, 0, 12, market(), 4, None, None,
                                                          record_actions=True)[0])
        self.assertEqual(actions.shape, (12, SCENARIO.num_zone, 3))
        np.testing.assert_allclose(actions[:5, :, 0], 1.1)
        np.testing.assert_allclose(actions[5:, :, 0], 1.0)
        np.testing.assert_allclose(actions[:, :, 2], 0.6, atol=1e-12)  # share follows p at the policy's share
        np.testing.assert_allclose(actions[:, :, 1], actions[:, :, 2] * actions[:, :, 0] / 0.75)

    def test_counterfactual_report_has_per_zone_actions(self):
        env = lib.make_environment(SCENARIO, market(), 11)
        arm = lib.Intervention("remove_drivers", 1, 10)
        report = evaluate(SCENARIO, lib.constant_policy(SCENARIO, 1.0, 0.6), market(), env.snapshot_state(), None, 0,
                          [arm], 2, 50, 1, horizons=[5, 10])
        cell = report[arm.label()]["10"]["actions"]
        for name in ("p", "q", "s"):
            self.assertEqual(len(cell[name]["diff"]), SCENARIO.num_zone)
            self.assertEqual(len(cell[name]["ci95"]), SCENARIO.num_zone)
            self.assertIn("zone_mean", cell[name])
        np.testing.assert_allclose(cell["p"]["diff"], 0, atol=1e-12)  # removing drivers leaves a constant policy alone
        self.assertIn("Δp=", format_table(report))


if __name__ == "__main__":
    unittest.main()
