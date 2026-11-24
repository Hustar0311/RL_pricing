"""Evaluate a legacy price_only TD3 checkpoint (main.py without --action_mode) under the rollout_lib protocol.

The legacy controller observes the environment through ``Platform.update_price`` (runner.py arguments); the policy
returns price p with order pay q = p, i.e. a driver share equal to delta, which is exactly the legacy settlement.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from center import Platform, observation_normalization  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402


class LegacyPolicy(lib.Policy):
    def decide(self, env, t):
        import torch
        with torch.no_grad():
            self.controller.update_price(env.pass_count, env.veh_count, env.get_ongoing_veh(), env.avg_profit,
                                         env.zone_profit, env.pricing_multipliers[0:max(5 // self.frequency, 1), :],
                                         t, "test", od_permutation=self.od_permutation, policy_constr=False,
                                         last_pricing=env.pricing_multipliers[0, :])
        return self.spec.decode(self.controller.prices.numpy())


def load_legacy_policy(model_dir, epoch, scenario, position_encode=64, policy_delay=0, actor_lr=1e-5, critic_lr=1e-3):
    with contextlib.redirect_stdout(io.StringIO()):
        controller = Platform(scenario.td, scenario.tt, scenario.num_zone, option="TD3_MLP", actor_lr=actor_lr,
            critic_lr=critic_lr, device="cpu", update_freq=1, veh_num=len(scenario.vp),
            permutation=scenario.permutation, permutation2=scenario.permutation2, od_permutation=True,
            searching="Gaussian", policy_delay=policy_delay, position_encode=position_encode,
            auto_correlation=False, forget=False, T_train=scenario.dd_train.shape[0],
            **observation_normalization(scenario.dd_train, scenario.vp))
    controller.pricer.load_weights(str(model_dir).rstrip("/") + "/", epoch)
    controller.pricer.eval()
    return LegacyPolicy(controller, 1, True)
