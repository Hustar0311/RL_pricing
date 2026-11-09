import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from center import Platform
from sim import Environment
from test_joint_contract import config


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def platform(option="TD3_MLP", frequency=1, mode="joint", device="cpu", permutation=True):
    folder = ROOT / "grid_small_static" / "4" / "4"
    td, tt, vp = [np.load(folder / (name + ".npy")) for name in ["td", "tt", "vp"]]
    with contextlib.redirect_stdout(io.StringIO()):
        controller = Platform(td, tt, 16, option=option, action_mode=mode, device=device,
            update_freq=frequency, T_train=1440, position_encode=8, kernel_size=3, stride=2, pooling=2,
            permutation=np.load(folder / "permutation.npy"), permutation2=np.load(folder / "permutation2.npy"),
            od_permutation=permutation, veh_num=len(vp), policy_delay=1)
        controller.pricer.initialize(torch.Generator().manual_seed(47))
    if device == "cuda":
        controller.pricer.cuda()
    return controller, Environment(td, tt, vp, 16, frequency=frequency, br_config=config(True, True))


class JointTD3Tests(unittest.TestCase):
    def test_network_update_and_executed_action_replay(self):
        for option in ["TD3_MLP", "TD3_CNN"]:
            for frequency in [1, 10]:
                for permute in ([True, False] if option.endswith("CNN") else [True]):
                    with self.subTest(option=option, frequency=frequency, permutation=permute):
                        random.seed(47)
                        controller, env = platform(option, frequency, permutation=permute)
                        state, state2d = controller.observe(env, permute)
                        before_actor = [p.detach().clone() for p in controller.pricer.actor.parameters()]
                        before_critic = [p.detach().clone() for p in controller.pricer.critic.parameters()]
                        for t in range(12):
                            action = controller.update_action(env, t, "train", permute)
                            self.assertEqual(action.price.shape, (16,))
                            np.testing.assert_allclose(controller.action_spec.decode(controller.prices.numpy()).compensation, action.compensation)
                            self.assertTrue(torch.all(controller.prices.abs() <= 1))
                            controller.buffer.push(state, state2d, controller.prices, state, state2d, float(t),
                                                   t, t + frequency, terminal=t == 11)
                        for iteration in range(6):
                            controller.pricer.update_policy(8, controller.buffer, iteration)
                        self.assertTrue(any(not torch.equal(a, b) for a,b in zip(before_actor, controller.pricer.actor.parameters())))
                        self.assertTrue(any(not torch.equal(a, b) for a,b in zip(before_critic, controller.pricer.critic.parameters())))
                        self.assertTrue(all(np.isfinite(v) for v in controller.pricer.last_metrics.values()))
                        batch = controller.buffer.sample(8)
                        self.assertEqual(batch[2].shape, (8, 32))
                        torch.testing.assert_close(batch[7] - batch[6], torch.full((8, 1), frequency))

    def test_replay_roundtrip_and_dimension_rejection(self):
        controller, env = platform()
        state, image = controller.observe(env)
        controller.update_action(env, 0, "test")
        controller.buffer.push(state, image, controller.prices, state, image, 2, 0, 10)
        with tempfile.TemporaryDirectory() as folder:
            controller.buffer.save(folder, 1)
            controller.buffer.clear()
            controller.buffer.load(folder, 1)
            self.assertEqual(len(controller.buffer), 1)
            controller.buffer.action_dim = 16
            with self.assertRaises(ValueError):
                controller.buffer.load(folder, 1)

    def test_legacy_network_shape_remains_loadable(self):
        for option in ["TD3_MLP", "TD3_CNN"]:
            old, env = platform(option, mode="price_only")
            new, _ = platform(option, mode="price_only")
            new.pricer.actor.load_state_dict(old.pricer.actor.state_dict())
            self.assertEqual(new.pricer.n_action, 16)


if __name__ == "__main__":
    unittest.main()
