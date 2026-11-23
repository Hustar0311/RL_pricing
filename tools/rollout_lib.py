"""Common-random-number rollouts from a simulator snapshot (shared by the evaluation tools).

Every arm of a comparison restores the same snapshot and replays the same minute-indexed random
streams (``Environment.reseed_streams``), so differences between arms are paired differences.
The environment must run with ``env_rng='isolated'``.
"""
import contextlib
import io
import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from center import Platform  # noqa: E402
from economic_action import ActionSpec, EconomicAction, SHARE_MODES  # noqa: E402
from joint_checkpoint import load_policy_only  # noqa: E402
from runner import _prepare_demand_batch, _sample_poisson_counts  # noqa: E402
from sampling import sample_poisson_counts  # noqa: E402
from sim import Environment  # noqa: E402
from welfare import step_welfare  # noqa: E402

METRICS = ["served", "revenue", "net_profit", "driver_income_signal", "online_vehicles", "entered",
           "driver_total_income", "consumer_surplus", "negative_margin_orders", "switches",
           "left", "wait_minutes", "pickup_minutes", "empty_minutes", "trip_distance", "trip_minutes", "driver_idle_cost"]
# metrics that accumulate over minutes (the others are read at the horizon minute)
WARMUP_OFFSET = 10_000_000  # warm-up minutes use step ids disjoint from the evaluation minutes
CUMULATIVE = {"served", "revenue", "net_profit", "entered", "driver_total_income", "consumer_surplus",
              "negative_margin_orders", "switches", "left", "wait_minutes", "pickup_minutes", "empty_minutes",
              "trip_distance", "trip_minutes", "driver_idle_cost"}


# ----------------------------------------------------------------------------- scenario / policy
class Scenario:
    def __init__(self, folder):
        self.folder = str(folder)
        load = lambda name: np.load(Path(folder) / (name + ".npy"))
        self.td, self.tt, self.vp = load("td"), load("tt"), load("vp")
        self.dd_train, self.dd_test = load("dd_train"), load("dd_test")
        self.permutation, self.permutation2 = load("permutation"), load("permutation2")
        self.num_zone = self.td.shape[0]


def market_namespace(configuration, overrides=None):
    """Environment configuration: checkpoint market rules, then explicit overrides."""
    values = dict(use_passenger_entry_br=False, use_driver_online_br=False, beta_passenger=1.0, beta_driver=1.0,
                  passenger_base_utility=2.0, passenger_price_sensitivity=1.0, passenger_wait_sensitivity=0.1,
                  passenger_calibration="none", target_demand_elasticity=-1.0, passenger_anchor_utility=1.0,
                  passenger_anchor_price=1.0, driver_decision_interval=1, driver_switch_cost=0.0, online_zone_choice="random", idle_cost_bearer="platform",
                  pricing_mode=False, frequency=1, seed=47, br_random_seed=None, env_rng="isolated")
    values.update({k: v for k, v in (configuration or {}).items() if k in values and v is not None})
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    values["env_rng"] = "isolated"
    values["use_bounded_rationality"] = bool(values["use_passenger_entry_br"] or values["use_driver_online_br"])
    return Namespace(**values)


def action_kwargs(meta):
    mode = meta["mode"]
    kwargs = dict(action_mode=mode)
    if mode == "joint":
        kwargs.update(compensation_min=meta["compensation_bounds"][0], compensation_max=meta["compensation_bounds"][1])
    if mode in SHARE_MODES:
        kwargs.update(share_min=meta["share_bounds"][0], share_max=meta["share_bounds"][1])
        if meta.get("share_value") is not None:
            kwargs.update(share_value=meta["share_value"])
    return kwargs


class Policy:
    """A decision rule with a held action between decisions; ``decide`` returns an EconomicAction."""

    def __init__(self, controller, frequency, od_permutation, constant=None):
        self.controller, self.frequency, self.od_permutation = controller, frequency, od_permutation
        self.constant = constant
        self.spec = controller.action_spec
        self.held = None

    def decide(self, env, t):
        if self.constant is not None:
            price, share = self.constant(t) if callable(self.constant) else self.constant
            price = np.broadcast_to(np.asarray(price, dtype=float), (self.spec.num_zone,)).copy()
            share_vec = np.broadcast_to(np.asarray(share, dtype=float), (self.spec.num_zone,)).copy()
            return EconomicAction(price, price * (share_vec / self.spec.delta), share_vec)
        with torch.no_grad():
            return self.controller.update_action(env, t, "test", self.od_permutation)

    def held_prices(self):
        return None if self.constant is not None else self.controller.prices.clone()

    def set_held_prices(self, prices):
        if self.constant is None and prices is not None:
            self.controller.prices = prices.clone()

    def action_from_prices(self, prices):
        return self.spec.decode(prices.numpy())


def load_policy(checkpoint_dir, epoch, scenario, device="cpu"):
    path = Path(checkpoint_dir) / "joint_checkpoint_{}.pt".format(epoch)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    contract = payload["contract"]
    config, meta = contract["configuration"], contract["action_space"]
    if meta["num_zone"] != scenario.num_zone:
        raise ValueError("policy was trained for {} zones, scenario has {}".format(meta["num_zone"], scenario.num_zone))
    with contextlib.redirect_stdout(io.StringIO()):
        controller = Platform(scenario.td, scenario.tt, scenario.num_zone, option=config["pricing_alg"],
            actor_lr=config["actor_lr"], critic_lr=config["critic_lr"], device=device, update_freq=config["frequency"],
            veh_num=len(scenario.vp), permutation=scenario.permutation, permutation2=scenario.permutation2,
            od_permutation=config["od_permutation"], kernel_size=config["kernel_size"], stride=config["stride"],
            pooling=config["pooling"], searching=config["searching"], policy_delay=config["policy_delay"],
            position_encode=config["position_encode"], auto_correlation=config["auto_correlation"],
            forget=config["forget"], T_train=contract["time_period"], **action_kwargs(meta))
    del payload
    load_policy_only(controller, path, device)
    return Policy(controller, config["frequency"], config["od_permutation"]), contract


def constant_policy(scenario, price=1.0, share=0.75, frequency=1):
    """price/share may be scalars, per-zone vectors, or a callable t -> (price, share) (time-varying schedules)."""
    with contextlib.redirect_stdout(io.StringIO()):
        controller = Platform(scenario.td, scenario.tt, scenario.num_zone, option="TD3_MLP", action_mode="fixed_share",
            share_value=0.75, update_freq=frequency, veh_num=len(scenario.vp), position_encode=8, T_train=1440,
            permutation=scenario.permutation, permutation2=scenario.permutation2, od_permutation=True)
    return Policy(controller, frequency, True, constant=(price, share))


def make_environment(scenario, market, seed, frequency=1):
    config = Namespace(**vars(market))
    config.seed, config.br_random_seed = seed, seed
    config.action_mode = "joint"  # settle with explicit compensation vectors and the order-compensation income signal
    return Environment(scenario.td, scenario.tt, scenario.vp, scenario.num_zone, frequency=frequency, br_config=config)


# ----------------------------------------------------------------------------- interventions
class Intervention:
    """price_multiply / price_set / share_set / share_shift / comp_multiply / demand_shock / remove_drivers."""
    KINDS = {"price_multiply", "price_set", "share_set", "share_shift", "comp_multiply", "demand_shock", "remove_drivers"}

    def __init__(self, kind, value, minutes=None, zones=None, hold_compensation=False):
        if kind not in self.KINDS:
            raise ValueError("unknown intervention " + kind)
        self.kind, self.value, self.minutes = kind, float(value), minutes
        self.zones, self.hold_compensation = zones, hold_compensation

    @classmethod
    def parse(cls, text, default_minutes, zones=None):
        body, _, minutes = text.partition("@")
        kind, _, value = body.partition(":")
        return cls(kind, value, int(minutes) if minutes else default_minutes, zones)

    def label(self):
        return "{}:{:g}@{}".format(self.kind, self.value, self.minutes)

    def active(self, minute):
        return self.minutes is None or minute < self.minutes

    def _mask(self, n):
        mask = np.zeros(n, dtype=bool)
        mask[list(self.zones) if self.zones else slice(None)] = True
        return mask

    def apply(self, action, delta, minute):
        """Return (price, compensation) after the intervention for this minute."""
        price, comp = action.price.copy(), action.compensation.copy()
        if not self.active(minute):
            return price, comp
        mask = self._mask(len(price))
        if self.kind in ("price_multiply", "price_set"):
            new_price = price.copy()
            new_price[mask] = price[mask] * self.value if self.kind == "price_multiply" else self.value
            if not self.hold_compensation:
                if action.share is not None:
                    comp = new_price * (action.share / delta)
                elif action.share is None and np.array_equal(comp, price):  # price_only: q follows p
                    comp = new_price.copy()
            price = new_price
        elif self.kind in ("share_set", "share_shift"):
            current = comp * delta / price  # implied current share
            target = current.copy()
            target[mask] = self.value if self.kind == "share_set" else current[mask] + self.value
            comp = price * (target / delta)
        elif self.kind == "comp_multiply":
            comp = comp.copy()
            comp[mask] = comp[mask] * self.value
        return price, comp

    def demand_factor(self, minute):
        return 1.0 + self.value if self.kind == "demand_shock" and self.active(minute) else 1.0

    def remove_drivers(self, env, seed):
        """Take k drivers out for good: idle ones go offline now, busy ones when they finish."""
        if self.kind != "remove_drivers":
            return
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7919]))
        candidates = [v for v in env.veh_list if v.state != 2]
        order = rng.permutation(len(candidates))
        for index in order[: int(self.value)]:
            v = candidates[index]
            v.profile = 1e9  # never goes online again
            if v.state == 0:
                env.veh_count[v.loc] -= 1
                env.veh_queue[v.loc].remove(v)
                env.active_veh -= 1
                v.state = 2
                v.trace_movement_type = "offline"


# ----------------------------------------------------------------------------- simulation
def simulate(env, policy, test_counts, t0, minutes, market, seed, intervention=None, held=None):
    """Run ``minutes`` steps from the restored state; returns an array (minutes, len(METRICS))."""
    spec = policy.spec
    policy.set_held_prices(held)
    action = policy.action_from_prices(held) if held is not None else None
    out = np.zeros((minutes, len(METRICS)))
    if intervention is not None:
        intervention.remove_drivers(env, seed)
    for k in range(minutes):
        t = t0 + k
        env.reseed_streams(seed, t)
        if action is None or t % policy.frequency == 0:
            action = policy.decide(env, t)
        price, comp = (action.price, action.compensation) if intervention is None else intervention.apply(action, spec.delta, k)
        factor = 1.0 if intervention is None else intervention.demand_factor(k)
        demand_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(t), 99]))
        demand = _prepare_demand_batch((test_counts[t:t + 1] * factor), price, market, demand_rng)[0]
        _, schedule = policy.controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)), env.veh_count)
        actual = env.apply_passenger_bounded_rationality(demand, price)
        result = env.step(actual, schedule, price, comp)
        revenue, expense, served, left = result[:4]
        stats, welfare, audit = env.get_last_step_stats(), step_welfare(env), env.get_last_step_audit()
        pickup = float(np.sum(audit["travel_time_matrix"][audit["service_vehicle_origin"], audit["service_passenger_origin"]]))
        origin, dest = audit["service_passenger_origin"], audit["service_destination"]
        trip_distance = float(np.sum(audit["travel_distance_matrix"][origin, dest]))
        trip_minutes = float(np.sum(audit["travel_time_matrix"][origin, dest]))
        out[k] = [served, revenue, revenue - expense, env.avg_profit, env.active_veh, stats["actual_passenger_count"],
                  welfare["driver_total_income"], welfare["consumer_surplus"], welfare["negative_margin_order_count"],
                  audit["driver_switch_count"], left, float(np.sum(audit["service_wait_index"])), pickup, result[7],
                  trip_distance, trip_minutes, welfare["driver_idle_cost"]]
    return out


def horizon_values(trace, horizons):
    """Cumulative metrics up to each horizon, level metrics at the horizon minute."""
    rows = {}
    for h in horizons:
        h = min(h, len(trace))
        values = {}
        for j, name in enumerate(METRICS):
            values[name] = float(trace[:h, j].sum()) if name in CUMULATIVE else float(trace[h - 1, j])
        rows[h] = values
    return rows


def confidence_interval(values, level=0.95):
    from scipy import stats
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan"), float("nan")
    mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    half = stats.t.ppf(0.5 + level / 2, len(values) - 1) * se
    return float(mean), float(mean - half), float(mean + half)


# ----------------------------------------------------------------------------- start state
def load_snapshot(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["env"], payload["prices"], int(payload["t"]), payload


def build_start_state(scenario, policy, market, start_minute, seed, warmup_minutes=1440):
    """Warm up on the training counts, then follow the policy through the test day up to start_minute."""
    env = make_environment(scenario, market, seed, policy.frequency)
    warm_start = max(0, len(scenario.dd_train) - warmup_minutes)
    action = None
    for t in range(warm_start, len(scenario.dd_train)):
        env.reseed_streams(seed, WARMUP_OFFSET + t)
        if action is None or t % policy.frequency == 0:
            action = policy.decide(env, t)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), WARMUP_OFFSET + t, 98]))
        demand = _prepare_demand_batch(sample_poisson_counts(scenario.dd_train[t][None], rng), action.price, market, rng)[0]
        _, schedule = policy.controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)), env.veh_count)
        env.step(env.apply_passenger_bounded_rationality(demand, action.price), schedule, action.price, action.compensation)
    trace_prices = None
    action = None
    for t in range(start_minute):
        env.reseed_streams(seed, t)
        if action is None or t % policy.frequency == 0:
            action = policy.decide(env, t)
        demand = _prepare_demand_batch(scenario.dd_test[t:t + 1], action.price, market, np.random.default_rng(np.random.SeedSequence([int(seed), int(t), 99])))[0]
        _, schedule = policy.controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)), env.veh_count)
        env.step(env.apply_passenger_bounded_rationality(demand, action.price), schedule, action.price, action.compensation)
    return env.snapshot_state(), policy.held_prices(), start_minute
