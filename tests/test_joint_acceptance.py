import contextlib
import io
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_checkpoint import load_checkpoint, save_checkpoint
from main import get_arguments
import test_joint_td3 as helpers


class JointAcceptanceTests(unittest.TestCase):
    def test_acceptance_device_names_do_not_collide(self):
        from tools.validate_joint_control import case_name
        cpu = ("dynamic", "TD3_CNN", 10, "cpu")
        gpu = ("dynamic", "TD3_CNN", 10, "cuda")
        self.assertNotEqual(case_name(cpu, "regression"), case_name(gpu, "regression"))

    def test_oldest_queued_control_is_observed_even_at_long_frequency(self):
        controller, env = helpers.platform(frequency=10)
        state, _ = controller.observe(env)
        env.compensation_multipliers[-1, 0] = 2.
        changed, _ = controller.observe(env)
        self.assertFalse(torch.equal(state, changed))
        env.pricing_multipliers[-1, 0] = 3.
        changed_price, _ = controller.observe(env)
        self.assertFalse(torch.equal(changed, changed_price))

    def test_unversioned_pinned_price_only_weights_still_load(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(["git", "show", "2424174:model.py"], cwd=root, text=True)
        original = types.ModuleType("pre_joint_model")
        exec(compile(source, "2424174:model.py", "exec"), original.__dict__)
        actor = original.ActorMLP(16, 5, 5, 7, [128, 128, 16], n=8, max_len=1440)
        critic = original.TwinCriticMLP(16, 5, 5, 7, [128, 128, 1], n=8, max_len=1440)
        with contextlib.redirect_stdout(io.StringIO()):
            actor.initialize(torch.Generator().manual_seed(47))
            critic.initialize(torch.Generator().manual_seed(47))
        controller, env = helpers.platform(mode="price_only")
        with tempfile.TemporaryDirectory() as folder:
            for name, module in [("actor", actor), ("actor_target", actor), ("critic", critic), ("critic_target", critic),
                                 ("actor_optim", torch.optim.Adam(actor.parameters())),
                                 ("critic_optim", torch.optim.Adam(critic.parameters()))]:
                torch.save(module.state_dict(), str(Path(folder)/(name+"_1.pkl")))
            controller.pricer.load_weights(folder+"/", 1)
            state, image = controller.observe(env)
            torch.testing.assert_close(controller.pricer.actor(state[None], image[None], torch.tensor([[0]])),
                                       actor(state[None], image[None], torch.tensor([[0]])), atol=0, rtol=0)
            joint, _ = helpers.platform()
            with self.assertRaisesRegex(ValueError, 'versioned joint'):
                joint.pricer.load_weights(folder+"/", 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_checkpoint_cpu_cuda_training_and_return_to_cpu(self):
        args = get_arguments(["--action_mode", "joint", "--pricing_alg", "TD3_MLP", "--position_encode", "8",
                              "--policy_delay", "1", "--use_bounded_rationality"])
        controller, env = helpers.platform()
        state, image = controller.observe(env)
        for t in range(8):
            controller.update_action(env, t, "train")
            controller.buffer.push(state, image, controller.prices, state, image, float(t), t, t+1)
        with tempfile.TemporaryDirectory() as folder:
            args.store_model_folder = folder
            save_checkpoint(controller, env, args, 1)
            gpu, gpu_env = helpers.platform()
            args.device = "cuda"
            load_checkpoint(gpu, gpu_env, args, 1)
            self.assertTrue(next(gpu.pricer.actor.parameters()).is_cuda)
            for iteration in range(6):
                gpu.pricer.update_policy(4, gpu.buffer, iteration)
            self.assertGreater(gpu.pricer.update_time, 0)
            self.assertTrue(all(np.isfinite(v) for v in gpu.pricer.last_metrics.values()))
            save_checkpoint(gpu, gpu_env, args, 2)
            cpu, cpu_env = helpers.platform()
            args.device = "cpu"
            load_checkpoint(cpu, cpu_env, args, 2)
            for left, right in zip(cpu.pricer.actor.parameters(), gpu.pricer.actor.parameters()):
                torch.testing.assert_close(left, right.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
