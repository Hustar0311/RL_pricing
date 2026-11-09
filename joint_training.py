"""Training and evaluation with aligned joint economic decisions."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from joint_checkpoint import load_checkpoint, save_checkpoint
from runner import (BR_LOG_COLUMNS, _accumulate_br_stats, _finalize_br_stats,
                    _prepare_demand_batch, _sample_poisson_counts, _zero_br_stats)
from sim import Environment


def _advance(env, controller, rate, action, args):
    _, schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)), env.veh_count)
    potential = _sample_poisson_counts(rate)
    demands = _prepare_demand_batch(potential[None], action.price, args)[0]
    actual = env.apply_passenger_bounded_rationality(demands, action.price)
    return env.step(actual, schedule, action.price, action.compensation)


def _warmup(env, controller, rates, args, mode):
    start = max(0, len(rates) - 1440)
    action = None
    for t in range(start, len(rates)):
        if action is None or t % args.frequency == 0:
            action = controller.update_action(env, t, mode, args.od_permutation)
        _advance(env, controller, rates[t], action, args)


def _parameter_vector(module):
    return torch.cat([p.detach().cpu().reshape(-1) for p in module.parameters()])


def train_joint(env, controller, rates, writer, args):
    if args.resume:
        load_checkpoint(controller, env, args, args.resume)
    else:
        controller.pricer.initialize(torch.Generator().manual_seed(args.seed))
        if args.device == "cuda":
            controller.pricer.cuda()
        _warmup(env, controller, rates, args, "train")
    controller.pricer.actor.train()
    controller.pricer.critic.train()
    history = []
    previous = Path(args.store_res_folder) / "train_log_{}.csv".format(args.resume)
    if args.resume and previous.exists():
        history = pd.read_csv(previous).to_dict("records")
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    start_clock = time.perf_counter()
    baseline = (1 - env.delta) * np.sum(rates * (env.alpha0 + env.alpha1 * env.travel_distance
                    + env.beta1 * env.travel_time)[None], axis=(1, 2))
    diagnostics = []
    for epoch in range(args.resume, args.n_epochs):
        epoch_clock = time.perf_counter()
        actor_before, critic_before = _parameter_vector(controller.pricer.actor), _parameter_vector(controller.pricer.critic)
        totals, br_totals = np.zeros(8), _zero_br_stats()
        active_total, learning_total = 0.0, 0.0
        state = image = action_tensor = None
        interval_reward = 0.0
        decision_t = 0
        for t in range(len(rates)):
            absolute_t = epoch * len(rates) + t
            if t % args.frequency == 0:
                state, image = controller.observe(env, args.od_permutation)
                action = controller.update_action(env, absolute_t, "train", args.od_permutation)
                action_tensor = controller.prices.clone()
                decision_t, interval_reward = absolute_t, 0.0
            outcome = _advance(env, controller, rates[t], action, args)
            totals += np.asarray(outcome)
            active_total += env.active_veh
            _accumulate_br_stats(br_totals, env.get_last_step_stats())
            learning_reward = outcome[0] - outcome[1] - baseline[t]
            interval_reward += learning_reward
            learning_total += learning_reward
            last_step = t + 1 == len(rates)
            if (t + 1) % args.frequency == 0 or last_step:
                terminal = last_step and env.active_veh == 0
                if terminal:
                    env.reset()
                next_state, next_image = controller.observe(env, args.od_permutation)
                controller.buffer.push(state, image, action_tensor, next_state, next_image,
                    interval_reward, decision_t, absolute_t + 1, terminal)
                if len(controller.buffer) >= args.batch_size:
                    controller.pricer.update_policy(args.batch_size, controller.buffer,
                        controller.pricer.critic_updates, update_freq=args.update_frequency)
        controller.decay_searching_variance()
        elapsed = previous_elapsed + time.perf_counter() - start_clock
        duration = time.perf_counter() - epoch_clock
        row = dict(zip(["income", "expense", "served", "left", "occu_mile", "occu_minute", "empty_mile", "empty_minute"], totals))
        row.update(epoch=epoch, reward=totals[0]-totals[1], learning_reward=learning_total,
                   act_veh=active_total/len(rates), elapsed_seconds=elapsed,
                   epoch_elapsed_seconds=duration, avg_epoch_seconds=elapsed/(epoch+1))
        row.update(_finalize_br_stats(br_totals))
        history.append(row)
        metrics = dict(controller.pricer.last_metrics)
        metrics.update(epoch=epoch, critic_updates=controller.pricer.critic_updates,
                       actor_updates=controller.pricer.update_time, replay_size=len(controller.buffer),
                       actor_parameter_change=float(torch.linalg.vector_norm(_parameter_vector(controller.pricer.actor)-actor_before)),
                       critic_parameter_change=float(torch.linalg.vector_norm(_parameter_vector(controller.pricer.critic)-critic_before)),
                       epsilon=controller.epsilon)
        diagnostics.append(metrics)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("non-finite joint training diagnostic")
        if writer is not None:
            writer.add_scalar("Reward/train", row["reward"], epoch)
            writer.flush()
        print(json.dumps({"completed_epoch": epoch+1, "reward": row["reward"], **metrics}), flush=True)
        save_checkpoint(controller, env, args, epoch+1)
        pd.DataFrame(history).to_csv(Path(args.store_res_folder) / "train_log_{}.csv".format(epoch+1), index=False)
        (Path(args.store_res_folder) / "training_diagnostics_{}.json".format(epoch+1)).write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8")


def evaluate_joint(env, controller, train_rates, test_rates, args):
    load_checkpoint(controller, env, args, args.n_epochs, training=False)
    controller.pricer.eval()
    repeats = [args.trace_test_k] if args.enable_visual_trace and args.trace_only and args.trace_test_k > 0 else range(1, 6)
    for repeat in repeats:
        seed = args.seed + repeat
        np.random.seed(seed)
        random.seed(seed)
        config = copy.copy(args)
        config.br_random_seed = (args.br_random_seed if args.br_random_seed is not None else args.seed) + repeat
        current = Environment(env.travel_distance, env.travel_time, env.veh_profiles, env.num_zone,
                              frequency=args.frequency, br_config=config)
        _warmup(current, controller, train_rates, args, "test")
        rows = []
        action = None
        for t in range(len(test_rates)):
            if action is None or t % args.frequency == 0:
                action = controller.update_action(current, t, "test", args.od_permutation)
            # Held-out demand is an observed count array, matching the legacy evaluator.
            _, schedule = controller.batch_matching(np.sum(current.pass_count, axis=(0, 2)), current.veh_count)
            demand = _prepare_demand_batch(test_rates[t:t+1], action.price, args)[0]
            demand = current.apply_passenger_bounded_rationality(demand, action.price)
            outcome = current.step(demand, schedule, action.price, action.compensation)
            row = dict(zip(["profit", "expense", "served", "left", "occu_mile", "occu_minute", "empty_mile", "empty_minute"], outcome))
            row.update(t=t, policy=action.price.copy(), act_veh=current.active_veh,
                       reward=outcome[0]-outcome[1], price_mean=float(action.price.mean()),
                       price_min=float(action.price.min()), price_max=float(action.price.max()))
            row.update(current.get_last_step_stats())
            rows.append(row)
        target = Path(args.store_res_folder) / "test_log_{}.csv".format(repeat)
        if not args.trace_only or not target.exists():
            pd.DataFrame(rows).to_csv(target, index=False)
        print(json.dumps({"test_repeat": repeat, "steps": len(rows), "reward": sum(row["reward"] for row in rows)}), flush=True)
