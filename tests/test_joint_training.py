import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from economic_action import ActionSpec
from joint_checkpoint import load_checkpoint
from joint_training import train_joint
from main import get_arguments
from test_joint_td3 import platform


class JointTrainingTests(unittest.TestCase):
    def arguments(self, folder, epochs=3, resume=0):
        args = get_arguments(["--action_mode", "joint", "--pricing_alg", "TD3_MLP", "--frequency", "5",
            "--batch_size", "2", "--n_epochs", str(epochs), "--resume", str(resume), "--position_encode", "8",
            "--policy_delay", "1", "--use_bounded_rationality"])
        args.store_model_folder = str(Path(folder) / "models")
        args.store_res_folder = str(Path(folder) / "results")
        Path(args.store_model_folder).mkdir(exist_ok=True)
        Path(args.store_res_folder).mkdir(exist_ok=True)
        return args

    def run_training(self, args):
        random.seed(47)
        np.random.seed(47)
        torch.manual_seed(47)
        controller, env = platform(frequency=5)
        rates = np.ones((13, 16, 16)) * .01
        with contextlib.redirect_stdout(io.StringIO()):
            train_joint(env, controller, rates, None, args)
        return controller, env

    def test_epoch_resume_is_exact_and_short_final_intervals_are_retained(self):
        with tempfile.TemporaryDirectory() as whole_dir, tempfile.TemporaryDirectory() as split_dir:
            whole_args = self.arguments(whole_dir)
            whole, whole_env = self.run_training(whole_args)
            split_args = self.arguments(split_dir, epochs=1)
            self.run_training(split_args)
            split_args = self.arguments(split_dir, epochs=3, resume=1)
            resumed, resumed_env = self.run_training(split_args)
            for key in ["actor", "critic", "actor_target", "critic_target"]:
                for left, right in zip(getattr(whole.pricer, key).parameters(), getattr(resumed.pricer, key).parameters()):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(whole.pricer.critic_updates, resumed.pricer.critic_updates)
            self.assertGreater(whole.pricer.update_time, 0)
            self.assertEqual(len(whole.buffer), 9)
            self.assertEqual([r.next_t-r.t for r in whole.buffer.memory], [5, 5, 3]*3)
            self.assertEqual(whole.epsilon, resumed.epsilon)
            np.testing.assert_array_equal(whole_env.pass_count, resumed_env.pass_count)
            resumed.action_spec = ActionSpec(16, "joint", .4, 2.)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_checkpoint(resumed, resumed_env, split_args, 3)

    def test_cli_rejects_incompatible_modes_and_bounds(self):
        cases = [["--action_mode", "joint"],
                 ["--compensation_min", "0"],
                 ["--action_mode", "joint", "--pricing_alg", "TD3_MLP", "--policy_constr"]]
        for argv in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                get_arguments(argv)


if __name__ == "__main__":
    unittest.main()
