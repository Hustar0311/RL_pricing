"""Exact epoch-boundary continuation for the joint-control runner."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def contract(controller, args):
    keys = ["pricing_alg", "frequency", "position_encode", "od_permutation", "auto_correlation",
            "searching", "policy_delay", "forget", "seed", "br_random_seed", "data_folder",
            "actor_lr", "critic_lr", "batch_size", "update_frequency", "kernel_size", "stride", "pooling",
            "use_passenger_entry_br", "use_driver_online_br",
            "beta_passenger", "beta_driver", "passenger_base_utility", "passenger_price_sensitivity",
            "passenger_wait_sensitivity"]
    return {"action_space": controller.action_spec.metadata(), "time_period": controller.pricer.actor.max_len,
            "configuration": {key: getattr(args, key, None) for key in keys}}


def save_checkpoint(controller, env, args, epoch):
    pricer = controller.pricer
    payload = {
        "version": 1, "contract": contract(controller, args), "epoch": epoch,
        "models": {key: getattr(pricer, key).state_dict() for key in
                   ["actor", "actor_target", "critic", "critic_target", "actor_optim", "critic_optim"]},
        "learner": {key: getattr(pricer, key) for key in
                    ["update_time", "critic_updates", "policy_freq", "last_metrics"]},
        "epsilon": controller.epsilon, "noise": getattr(controller, "noise", None),
        "environment": env.__dict__, "replay": controller.buffer.memory,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if str(args.device) == "cuda" else []},
    }
    folder = Path(args.store_model_folder)
    torch.save(payload, folder / ("joint_checkpoint_{}.pt".format(epoch)))
    controller.pricer.save_model(str(folder) + "/", epoch)
    (folder / ("action_spec_{}.json".format(epoch))).write_text(
        json.dumps(payload["contract"], indent=2), encoding="utf-8")


def load_checkpoint(controller, env, args, epoch, training=True):
    path = Path(args.store_model_folder) / ("joint_checkpoint_{}.pt".format(epoch))
    if not path.exists():
        raise ValueError("joint checkpoint missing; a price-only checkpoint cannot be loaded into joint mode")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("contract") != contract(controller, args):
        raise ValueError("checkpoint action space, state layout or experiment configuration does not match")
    if payload.get("epoch") != epoch:
        raise ValueError("checkpoint epoch does not match requested resume")
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
