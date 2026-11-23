"""Exact epoch-boundary continuation for the joint-control runner."""
import json
import random
from pathlib import Path

import numpy as np
import torch


OPTIONAL_CONTRACT_DEFAULTS = {
    "env_rng": "legacy", "passenger_calibration": "none", "target_demand_elasticity": -1.0,
    "passenger_anchor_utility": 1.0, "passenger_anchor_price": 1.0,
    "driver_decision_interval": 1, "driver_switch_cost": 0.0,
    "online_zone_choice": "random", "idle_cost_bearer": "platform", "driver_signal": "gross", "share_lr": 1e-3, "share_init": None, "share_head_separate": False,
}


def contract(controller, args):
    keys = ["pricing_alg", "frequency", "position_encode", "od_permutation", "auto_correlation",
            "searching", "policy_delay", "forget", "seed", "br_random_seed", "data_folder",
            "actor_lr", "critic_lr", "batch_size", "update_frequency", "kernel_size", "stride", "pooling",
            "use_passenger_entry_br", "use_driver_online_br",
            "beta_passenger", "beta_driver", "passenger_base_utility", "passenger_price_sensitivity",
            "passenger_wait_sensitivity"]
    configuration = {key: getattr(args, key, None) for key in keys}
    # market-rule switches enter the contract only when they differ from the legacy default, so
    # checkpoints written before these options existed keep loading under default settings
    for key, default in OPTIONAL_CONTRACT_DEFAULTS.items():
        value = getattr(args, key, default)
        if value != default:
            configuration[key] = value
    return {"action_space": controller.action_spec.metadata(), "time_period": controller.pricer.actor.max_len,
            "configuration": configuration}


def keep_replay(args, epoch):
    """--save_replay last (default): only the final epoch carries the replay buffer; none: never."""
    mode = getattr(args, "save_replay", "last")
    return mode == "last" and epoch == args.n_epochs


def prune_checkpoints(folder, keep, newest):
    """Delete per-epoch model files older than the ``keep`` most recent epochs (<= 0 keeps everything)."""
    if keep <= 0:
        return
    import re
    for path in Path(folder).iterdir():
        match = re.search(r"_(\d+)\.(pt|pkl|json)$", path.name)
        if match and path.name.startswith(("joint_checkpoint", "actor", "critic", "action_spec")) \
                and int(match.group(1)) <= newest - keep:
            path.unlink()


def save_checkpoint(controller, env, args, epoch):
    pricer = controller.pricer
    payload = {
        "version": 1, "contract": contract(controller, args), "epoch": epoch,
        "models": {key: getattr(pricer, key).state_dict() for key in
                   ["actor", "actor_target", "critic", "critic_target", "actor_optim", "critic_optim"]},
        "learner": {key: getattr(pricer, key) for key in
                    ["update_time", "critic_updates", "policy_freq", "last_metrics"]},
        "epsilon": controller.epsilon, "noise": getattr(controller, "noise", None),
        "environment": env.__dict__, "replay": controller.buffer.memory if keep_replay(args, epoch) else None,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if str(args.device) == "cuda" else []},
    }
    folder = Path(args.store_model_folder)
    torch.save(payload, folder / ("joint_checkpoint_{}.pt".format(epoch)))
    controller.pricer.save_model(str(folder) + "/", epoch)
    (folder / ("action_spec_{}.json".format(epoch))).write_text(
        json.dumps(payload["contract"], indent=2), encoding="utf-8")
    prune_checkpoints(folder, getattr(args, "keep_checkpoints", 2), epoch)


def load_checkpoint(controller, env, args, epoch, training=True):
    path = Path(args.store_model_folder) / ("joint_checkpoint_{}.pt".format(epoch))
    if not path.exists():
        raise ValueError("joint checkpoint missing; a price-only checkpoint cannot be loaded into joint mode")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("contract") != contract(controller, args):
        raise ValueError("checkpoint action space, state layout or experiment configuration does not match")
    if payload.get("epoch") != epoch:
        raise ValueError("checkpoint epoch does not match requested resume")
    if training and payload.get("replay") is None:
        raise ValueError("checkpoint was saved without its replay buffer (--save_replay last/none); "
                         "only the final epoch of a --save_replay last run can be resumed")
    controller.device = args.device
    controller.buffer.device = args.device
    for name in ["actor", "actor_target", "critic", "critic_target"]:
        getattr(controller.pricer, name).to(args.device)
    for key, value in payload["models"].items():
        if training or not key.endswith("optim"):
            getattr(controller.pricer, key).load_state_dict(value)
    if training:
        for key, value in payload["learner"].items():
            setattr(controller.pricer, key, value)
        controller.epsilon = payload["epsilon"]
        if payload["noise"] is not None:
            controller.noise = payload["noise"]
        controller.buffer.memory = payload["replay"]
        env.__dict__.clear()
        env.__dict__.update(payload["environment"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if args.device == "cuda" and payload["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload


def load_policy_only(controller, path, device="cpu"):
    """Load actor/critic weights for offline evaluation under a different market.

    Only the action-space contract is enforced; training-time market rules (BR switches,
    calibration, driver inertia) are deliberately not compared because counterfactual
    evaluation changes them on purpose.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != 1:
        raise ValueError("unsupported joint checkpoint version")
    if payload["contract"]["action_space"] != controller.action_spec.metadata():
        raise ValueError("checkpoint action space does not match the evaluation controller")
    controller.device = device
    for name in ["actor", "actor_target", "critic", "critic_target"]:
        getattr(controller.pricer, name).to(device)
        getattr(controller.pricer, name).load_state_dict(payload["models"][name])
    controller.pricer.eval()
    return payload["contract"]
