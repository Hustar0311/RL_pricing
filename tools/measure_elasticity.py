"""Measure the arc elasticities of passenger entry (to price) and driver supply (to compensation).

Loads one policy and one simulator state and replays it N times with common random numbers:
  * epsilon_D: p scaled by 1 +/- 5% for ``--demand_window`` minutes (q held fixed), response = entering passengers;
  * epsilon_S: q scaled by 1 +/- 5% for ``--supply_window`` minutes (p held fixed), response = online vehicles
    (mean over the last ``--supply_tail`` minutes of the window, and the end-point value).
Arc elasticity per replicate: ((x+ - x-) / mean(x+, x-)) / ((f+ - f-) / mean(f+, f-)); output is the mean
over replicates with a 95% t-interval, plus the elasticity of the replicate-averaged responses.

The state comes from ``--snapshot`` (saved by main.py --snapshot_steps) or is built from ``--scenario``
(warm-up on the training counts followed by ``--start_minute`` test minutes).  Run from the worktree.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402


def arc_elasticity(plus, minus, factor_plus, factor_minus):
    mean = (plus + minus) / 2.0
    dlog_x = np.where(mean != 0, (plus - minus) / np.where(mean != 0, mean, 1.0), np.nan)
    dlog_f = (factor_plus - factor_minus) / ((factor_plus + factor_minus) / 2.0)
    return dlog_x / dlog_f


def summarize(per_replicate, plus_values, minus_values, factors):
    mean, low, high = lib.confidence_interval(per_replicate)
    pooled = float(arc_elasticity(np.mean(plus_values), np.mean(minus_values), *factors))
    return {"elasticity_mean": mean, "ci95": [low, high], "pooled_arc_elasticity": pooled,
            "response_plus_mean": float(np.mean(plus_values)), "response_minus_mean": float(np.mean(minus_values)),
            "replicates": int(np.isfinite(per_replicate).sum())}


def measure(scenario, policy, market, state, held, t0, replicates, seed0, delta, demand_window, supply_window,
            supply_tail, frequency):
    env = lib.make_environment(scenario, market, seed0, frequency)
    total = max(demand_window, supply_window)
    counts = scenario.dd_test
    if t0 + total > len(counts):
        raise ValueError("snapshot minute {} + window {} exceeds the test horizon {}".format(t0, total, len(counts)))
    arms = {
        "p-": lib.Intervention("price_multiply", 1 - delta, demand_window, hold_compensation=True),
        "p+": lib.Intervention("price_multiply", 1 + delta, demand_window, hold_compensation=True),
        "q-": lib.Intervention("comp_multiply", 1 - delta, supply_window),
        "q+": lib.Intervention("comp_multiply", 1 + delta, supply_window),
    }
    entered = {k: [] for k in arms}
    online_tail = {k: [] for k in arms}
    online_end = {k: [] for k in arms}
    online_rate = []
    for r in range(replicates):
        seed = seed0 + r
        for name, arm in arms.items():
            env.restore_state(state)
            trace = lib.simulate(env, policy, counts, t0, supply_window if name[0] == "q" else demand_window, market,
                                 seed, arm, held)
            col = lib.METRICS.index
            if name[0] == "p":
                entered[name].append(trace[:demand_window, col("entered")].sum())
            else:
                online_tail[name].append(trace[-supply_tail:, col("online_vehicles")].mean())
                online_end[name].append(trace[-1, col("online_vehicles")])
        env.restore_state(state)
        online_rate.append(lib.simulate(env, policy, counts, t0, supply_window, market, seed, None, held)[:, lib.METRICS.index("online_vehicles")].mean() / len(scenario.vp))
    factors = (1 + delta, 1 - delta)
    arr = lambda d, k: np.asarray(d[k], dtype=float)
    eps_d = arc_elasticity(arr(entered, "p+"), arr(entered, "p-"), *factors)
    eps_s_tail = arc_elasticity(arr(online_tail, "q+"), arr(online_tail, "q-"), *factors)
    eps_s_end = arc_elasticity(arr(online_end, "q+"), arr(online_end, "q-"), *factors)
    return {
        "epsilon_D": summarize(eps_d, arr(entered, "p+"), arr(entered, "p-"), factors),
        "epsilon_S_tail_mean": summarize(eps_s_tail, arr(online_tail, "q+"), arr(online_tail, "q-"), factors),
        "epsilon_S_endpoint": summarize(eps_s_end, arr(online_end, "q+"), arr(online_end, "q-"), factors),
        "baseline_online_rate_mean": float(np.mean(online_rate)),
        "baseline_online_rate_ci95": list(lib.confidence_interval(online_rate)[1:]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="data folder with td/tt/vp/dd_*/permutation*.npy")
    parser.add_argument("--checkpoint_dir", help="folder holding joint_checkpoint_<epoch>.pt")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--constant_policy", nargs=2, type=float, metavar=("PRICE", "SHARE"),
                        help="evaluate a constant price/share policy instead of a checkpoint")
    parser.add_argument("--snapshot", help="pickle written by main.py --snapshot_steps")
    parser.add_argument("--start_minute", type=int, default=360, help="fork minute when no snapshot is given")
    parser.add_argument("--replicates", "-N", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--perturbation", type=float, default=0.05)
    parser.add_argument("--demand_window", type=int, default=30)
    parser.add_argument("--supply_window", type=int, default=60)
    parser.add_argument("--supply_tail", type=int, default=10)
    parser.add_argument("--passenger_br", choices=["on", "off", "checkpoint"], default="checkpoint")
    parser.add_argument("--driver_br", choices=["on", "off", "checkpoint"], default="checkpoint")
    parser.add_argument("--market", default="", help="extra market overrides as key=value, comma separated")
    parser.add_argument("--output", help="write the result JSON here")
    args = parser.parse_args(argv)

    scenario = lib.Scenario(args.scenario)
    if args.constant_policy:
        policy, configuration = lib.constant_policy(scenario, *args.constant_policy), {}
    else:
        policy, contract = lib.load_policy(args.checkpoint_dir, args.epoch, scenario)
        configuration = contract["configuration"]
    overrides = lib_overrides(args)
    market = lib.market_namespace(configuration, overrides)
    if args.snapshot:
        state, held, t0, _ = lib.load_snapshot(args.snapshot)
    else:
        state, held, t0 = lib.build_start_state(scenario, policy, market, args.start_minute, args.seed - 1)
    result = measure(scenario, policy, market, state, held, t0, args.replicates, args.seed, args.perturbation,
                     args.demand_window, args.supply_window, args.supply_tail, policy.frequency)
    result["settings"] = {"fork_minute": t0, "replicates": args.replicates, "perturbation": args.perturbation,
                          "demand_window": args.demand_window, "supply_window": args.supply_window,
                          "scenario": args.scenario, "vehicles": int(len(scenario.vp)),
                          "market": {k: v for k, v in vars(market).items()}}
    text = json.dumps(result, indent=2, default=float)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return result


def lib_overrides(args):
    overrides = {}
    if args.passenger_br != "checkpoint":
        overrides["use_passenger_entry_br"] = args.passenger_br == "on"
    if args.driver_br != "checkpoint":
        overrides["use_driver_online_br"] = args.driver_br == "on"
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value) if value[:1] in "0123456789-.tfn[" else value
    return overrides


if __name__ == "__main__":
    main()
