import tempfile
import unittest
from pathlib import Path

import torch

from joint_checkpoint import load_checkpoint
from test_joint_training import JointTrainingTests
from test_joint_td3 import platform


class CheckpointRetentionTests(unittest.TestCase):
    def train(self, folder, epochs, *extra):
        helper = JointTrainingTests()
        args = helper.arguments(folder, epochs=epochs)
        for key, value in extra:
            setattr(args, key, value)
        helper.run_training(args)
        return args

    def epochs_on_disk(self, args):
        return sorted(int(p.stem.split("_")[-1]) for p in Path(args.store_model_folder).glob("joint_checkpoint_*.pt"))

    def test_keep_last_n_and_replay_only_in_final_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.train(folder, 4)  # defaults: keep_checkpoints=2, save_replay=last
            self.assertEqual(self.epochs_on_disk(args), [3, 4])
            self.assertEqual(sorted(int(p.stem.split("_")[-1]) for p in Path(args.store_model_folder).glob("actor_[0-9]*.pkl")), [3, 4])
            self.assertIsNone(torch.load(Path(args.store_model_folder) / "joint_checkpoint_3.pt", weights_only=False)["replay"])
            self.assertIsNotNone(torch.load(Path(args.store_model_folder) / "joint_checkpoint_4.pt", weights_only=False)["replay"])
            controller, env = platform(frequency=5)
            with self.assertRaisesRegex(ValueError, "replay"):
                load_checkpoint(controller, env, args, 3)

    def test_keep_all_and_no_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.train(folder, 3, ("keep_checkpoints", 0), ("save_replay", "none"))
            self.assertEqual(self.epochs_on_disk(args), [1, 2, 3])
            for epoch in [1, 3]:
                self.assertIsNone(torch.load(Path(args.store_model_folder) / "joint_checkpoint_{}.pt".format(epoch), weights_only=False)["replay"])


if __name__ == "__main__":
    unittest.main()
