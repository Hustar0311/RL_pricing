import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from economic_action import ActionSpec, action_dimension
from joint_checkpoint import load_checkpoint
from joint_training import train_joint
from main import get_arguments
from sim import Environment
from test_joint_td3 import ROOT, platform
from test_joint_training import JointTrainingTests

import random


def make_env(mode, seed=47):
    folder = ROOT / "grid_small_static" / "4" / "4"
    td, tt, vp = [np.load(folder / (name + ".npy")) for name in ["td", "tt", "vp"]]
    config = SimpleNamespace(action_mode=mode, use_passenger_entry_br=True, use_driver_online_br=True,
                             seed=seed, br_random_seed=seed)
    np.random.seed(seed)
    return Environment(td, tt, vp, 16, br_config=config)


def run_trajectory(mode, spec, normalized_sequence, controller, steps, demand_seed=5, force_q=None):
    """Drive one environment with a given normalized action sequence and return per-step records."""
    env = make_env(mode)
    demand_rng = np.random.default_rng(demand_seed)
    records = []
    for t in range(steps):
        action = spec.decode(normalized_sequence[t])
        potential = demand_rng.poisson(0.05, (16, 16))
        _, schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)), env.veh_count)
        actual = env.apply_passenger_bounded_rationality(potential, action.price)
        compensation = action.compensation if mode != "price_only" else None
        outcome = env.step(actual, schedule, action.price, compensation)
        audit = env.get_last_step_audit()
        records.append(dict(outcome=outcome, pass_count=env.pass_count.copy(), veh_count=env.veh_count.copy(),
                            avg_profit=env.avg_profit, zone_profit=env.zone_profit.copy(),
                            active=env.active_veh, payment=audit["driver_payment_total"],
                            basis=audit["driver_income_basis"]))
    return records


class ShareModeTests(unittest.TestCase):
    STEPS = 40

    def setUp(self):
        self.controller, _ = platform(mode="fixed_share")
        rng = np.random.default_rng(3)
        self.prices = rng.uniform(-1, 1, (self.STEPS, 16))

    def assertSameTrajectory(self, left, right):
        for a, b in zip(left, right):
            self.assertEqual(a["outcome"], b["outcome"])
            for key in ["pass_count", "veh_count", "zone_profit"]:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a["avg_profit"], b["avg_profit"])
            self.assertEqual(a["active"], b["active"])
            self.assertEqual(a["payment"], b["payment"])

    def test_fixed_share_at_delta_matches_price_only_step_by_step(self):
        price_only = run_trajectory("price_only", ActionSpec(16, "price_only"), self.prices, self.controller, self.STEPS)
        fixed = run_trajectory("fixed_share", ActionSpec(16, "fixed_share", share_value=0.75), self.prices,
                               self.controller, self.STEPS)
        self.assertGreater(sum(r["outcome"][2] for r in price_only), 0)
        self.assertSameTrajectory(price_only, fixed)

    def test_joint_share_at_constant_delta_matches_fixed_share(self):
        fixed = run_trajectory("fixed_share", ActionSpec(16, "fixed_share", share_value=0.75), self.prices,
                               self.controller, self.STEPS)
        # with bounds (0.5, 1.0) a zero share action decodes to exactly s = 0.75
        spec = ActionSpec(16, "joint_share", share_min=0.5, share_max=1.0)
        actions = np.concatenate([self.prices, np.zeros((self.STEPS, 16))], axis=1)
        np.testing.assert_array_equal(spec.decode(actions[0]).share, np.full(16, 0.75))
        joint_share = run_trajectory("joint_share", spec, actions, self.controller, self.STEPS)
        self.assertSameTrajectory(fixed, joint_share)
        # default bounds (0.2, 0.95): this x decodes to 0.75 up to rounding
        default = ActionSpec(16, "joint_share")
        np.testing.assert_allclose(default.decode(np.r_[self.prices[0], np.full(16, 2 * (0.75 - 0.2) / 0.75 - 1)]).share, 0.75)

    def test_driver_payment_is_share_of_passenger_revenue(self):
        for share in [0.65, 0.85]:
            spec = ActionSpec(16, "fixed_share", share_value=share)
            records = run_trajectory("fixed_share", spec, self.prices, self.controller, self.STEPS)
            revenue = sum(r["outcome"][0] for r in records)
            payment = sum(r["payment"] for r in records)
            self.assertGreater(revenue, 0)
            self.assertAlmostEqual(payment / revenue, share, places=9)

    def test_action_layout_bounds_and_dimensions(self):
        self.assertEqual([action_dimension(m, 16) for m in ["price_only", "fixed_share", "learned_share", "joint_share", "joint"]],
                         [16, 16, 17, 32, 32])
        learned = ActionSpec(2, "learned_share", share_min=0.45, share_max=0.95)
        extreme = learned.decode([-3, 3, -1])
        np.testing.assert_allclose(extreme.share, [0.45, 0.45])
        np.testing.assert_allclose(extreme.compensation, np.array([1 / 3, 3]) * 0.45 / 0.75)
        np.testing.assert_allclose(learned.decode([0, 0, 1]).share, 0.95)
        per_zone = ActionSpec(2, "joint_share").decode([0, 0, -1, 1])
        np.testing.assert_allclose(per_zone.share, [0.2, 0.95])
        with self.assertRaises(ValueError):
            learned.decode([0, 0])
        for bounds in [(0.9, 0.5), (0.0, 0.9), (0.5, np.inf)]:
            with self.assertRaises(ValueError):
                ActionSpec(2, "joint_share", share_min=bounds[0], share_max=bounds[1])

    def test_learned_share_depends_on_parameter_not_state(self):
        for option in ["TD3_MLP", "TD3_CNN"]:
            with self.subTest(option=option):
                controller, env = platform(option, mode="learned_share")
                actor = controller.pricer.actor
                state, state2d = controller.observe(env, True)
                state = state[None].clone().requires_grad_(True)
                state2d = state2d[None].clone().requires_grad_(True)
                t = torch.tensor([[3]])
                output = actor(state, state2d, t)
                self.assertEqual(output.shape, (1, 17))
                share_grads = torch.autograd.grad(output[:, -1].sum(), [state, state2d], allow_unused=True, retain_graph=True)
                for grad in share_grads:
                    self.assertTrue(grad is None or torch.count_nonzero(grad) == 0)
                actor.zero_grad()
                output[:, -1].sum().backward(retain_graph=True)
                self.assertGreater(float(actor.free_logit.grad.abs().sum()), 0)
                # price outputs remain state dependent after the (zero-initialised) output layer is perturbed
                perturbed = actor.linears[-1].weight.data
                perturbed += 0.1
                output = actor(state, state2d, t)
                price_grad = torch.autograd.grad(output[:, :16].sum(), state, retain_graph=True)[0]
                self.assertGreater(float(price_grad.abs().sum()), 0)
                # the share output is identical for different states
                other = actor(state.detach() * 0.5, state2d.detach(), t)
                self.assertEqual(float(other[0, -1]), float(output[0, -1]))

    def test_learned_share_enters_td3_update_and_exploration(self):
        controller, env = platform(mode="learned_share")
        controller.epsilon = 0.5
        state, state2d = controller.observe(env, True)
        shares = set()
        for t in range(8):
            action = controller.update_action(env, t, "train", True)
            shares.add(round(float(action.share[0]), 9))
            self.assertTrue(np.all(action.share == action.share[0]))
            controller.buffer.push(state, state2d, controller.prices, state, state2d, float(t), t, t + 1, False)
        self.assertGreater(len(shares), 1)  # exploration noise acts on the share entry
        before = controller.pricer.actor.free_logit.detach().clone()
        for _ in range(3):
            controller.pricer.policy_freq = 0
            controller.pricer.update_policy(4, controller.buffer, 0)
        self.assertFalse(torch.equal(before, controller.pricer.actor.free_logit.detach()))

    def test_cli_and_checkpoint_mode_mismatch(self):
        for mode in ["fixed_share", "learned_share", "joint_share"]:
            args = get_arguments(["--action_mode", mode, "--pricing_alg", "TD3_MLP"])
            self.assertEqual(args.share_min, 0.2)
            self.assertEqual(args.share_max, 0.95)
            self.assertEqual(args.share_value, 0.75)
        for argv in [["--action_mode", "fixed_share"], ["--action_mode", "joint_share", "--pricing_alg", "TD3_MLP", "--share_min", "0.96"],
                     ["--action_mode", "fixed_share", "--pricing_alg", "TD3_MLP", "--share_value", "0"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                get_arguments(argv)
        helper = JointTrainingTests()
        with tempfile.TemporaryDirectory() as folder:
            args = get_arguments(["--action_mode", "fixed_share", "--pricing_alg", "TD3_MLP", "--frequency", "5",
                                  "--batch_size", "2", "--n_epochs", "1", "--position_encode", "8", "--policy_delay", "1",
                                  "--use_bounded_rationality"])
            args.store_model_folder = str(Path(folder) / "models")
            args.store_res_folder = str(Path(folder) / "results")
            Path(args.store_model_folder).mkdir()
            Path(args.store_res_folder).mkdir()
            random.seed(47); np.random.seed(47); torch.manual_seed(47)
            controller, env = platform(frequency=5, mode="fixed_share")
            with contextlib.redirect_stdout(io.StringIO()):
                train_joint(env, controller, np.ones((13, 16, 16)) * .01, None, args)
            row = np.genfromtxt(Path(args.store_res_folder) / "train_log_1.csv", delimiter=",", names=True)
            self.assertAlmostEqual(float(row["share_mean"]), 0.75)
            # same mode reloads; any other mode or other share value is refused
            load_checkpoint(controller, env, args, 1)
            for mode, kwargs in [("joint_share", {}), ("learned_share", {}), ("fixed_share", {"share_value": 0.65}),
                                 ("joint", {})]:
                other, other_env = platform(frequency=5, mode=mode)
                from economic_action import ActionSpec as Spec
                if kwargs:
                    other.action_spec = Spec(16, mode, **kwargs)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_checkpoint(other, other_env, args, 1)


class ShareOptimizerTests(unittest.TestCase):
    def test_share_group_has_own_lr_and_is_not_clipped(self):
        controller, env = platform(mode="learned_share")
        pricer = controller.pricer
        self.assertEqual(len(pricer.actor_optim.param_groups), 1)  # default platform(): no share_lr given
        import contextlib, io
        from center import Platform
        folder = ROOT / "grid_small_static" / "4" / "4"
        td, tt = np.load(folder / "td.npy"), np.load(folder / "tt.npy")
        with contextlib.redirect_stdout(io.StringIO()):
            controller = Platform(td, tt, 16, option="TD3_MLP", action_mode="learned_share", T_train=1440, position_encode=8,
                                  share_lr=5e-3, share_init=0.4, policy_delay=1,
                                  permutation=np.load(folder / "permutation.npy"), permutation2=np.load(folder / "permutation2.npy"))
            controller.pricer.initialize(torch.Generator().manual_seed(1))
        pricer, spec = controller.pricer, controller.action_spec
        self.assertEqual([g["lr"] for g in pricer.actor_optim.param_groups], [1e-4, 5e-3])
        share = pricer.actor.share_parameters()
        self.assertEqual(len(share), 1)
        self.assertNotIn(id(share[0]), {id(p) for p in pricer._clipped_actor_parameters()})
        # initial deterministic share equals share_init
        unit = float(torch.tanh(pricer.actor.free_logit)[0])
        self.assertAlmostEqual(spec.share_min + (unit + 1) / 2 * (spec.share_max - spec.share_min), 0.4, places=6)
        self.assertEqual(float(torch.tanh(pricer.actor_target.free_logit)[0]), unit)
        with self.assertRaises(ValueError):
            Platform(td, tt, 16, option="TD3_MLP", action_mode="learned_share", T_train=1440, position_encode=8,
                     share_init=0.99, permutation=np.load(folder / "permutation.npy"), permutation2=np.load(folder / "permutation2.npy"))

    def test_share_gradient_is_unclipped_after_update(self):
        controller, env = platform(mode="learned_share")
        state, state2d = controller.observe(env, True)
        for t in range(8):
            controller.update_action(env, t, "train", True)
            controller.buffer.push(state, state2d, controller.prices, state, state2d, 1000.0 * (t % 3), t, t + 1, False)
        controller.pricer.policy_freq = 0
        controller.pricer.update_policy(8, controller.buffer, 0)
        grad = controller.pricer.actor.free_logit.grad
        total = torch.sqrt(sum((p.grad ** 2).sum() for p in controller.pricer._clipped_actor_parameters() if p.grad is not None))
        self.assertLessEqual(float(total), 0.0100001)  # the clipped group is bounded by MAX_GRAD_NORM_ACTOR
        self.assertIsNotNone(grad)

    def test_separate_joint_share_head(self):
        for option in ["TD3_MLP", "TD3_CNN"]:
            import contextlib, io
            from center import Platform
            folder = ROOT / "grid_small_static" / "4" / "4"
            td, tt = np.load(folder / "td.npy"), np.load(folder / "tt.npy")
            with contextlib.redirect_stdout(io.StringIO()):
                controller = Platform(td, tt, 16, option=option, action_mode="joint_share", T_train=1440, position_encode=8,
                                      kernel_size=3, stride=2, pooling=2, share_lr=2e-3, share_head_separate=True,
                                      permutation=np.load(folder / "permutation.npy"),
                                      permutation2=np.load(folder / "permutation2.npy"), od_permutation=True)
                controller.pricer.initialize(torch.Generator().manual_seed(1))
            actor = controller.pricer.actor
            self.assertEqual(len(actor.share_parameters()), 2)
            self.assertEqual([g["lr"] for g in controller.pricer.actor_optim.param_groups], [1e-4, 2e-3])
            state, state2d = torch.zeros(1, 1), torch.zeros(1)
            from test_joint_td3 import platform as make_platform
            c2, env = make_platform(option, mode="joint_share")
            s, s2 = c2.observe(env, True)
            output = c2.pricer.actor(s[None], s2[None], torch.tensor([[1]]))
            self.assertEqual(output.shape, (1, 32))
            self.assertEqual(c2.pricer.actor.share_parameters(), [])  # default platform: shared output layer


if __name__ == "__main__":
    unittest.main()
