"""Search (demand_scale x fleet_multiplier x passenger wait sensitivity) for a supply-binding scenario.

Constant price p = 1 and global share s in {0.2, 0.4, 0.6, 0.75, 0.9}; each cell is evaluated on the full test day
with common random numbers (warm-up on the training counts under the same policy).  Reported per cell: served/entered,
mean wait of served orders, mean pickup time, passengers who expired, online rate, driver hourly wage and the arc
elasticity of served orders with respect to s (between s=0.6 and s=0.9, per replicate, with a 95% t-interval).
Selection: served/entered at s=0.75 in [lo, hi] and the elasticity interval strictly above 0.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402
from tools.make_scenario import SCENARIOS, build  # noqa: E402

SHARES = [0.2, 0.4, 0.6, 0.75, 0.9]
DEMAND = [1.5, 2.0, 3.0]
FLEET = [1.0, 1.5, 2.0, 3.0]
GAMMA = [0.1, 0.5]
MARKET = dict(use_passenger_entry_br=True, use_driver_online_br=True, passenger_calibration="anchor",
              driver_decision_interval=30)
_CTX = {}


def cell_metrics(trace, vehicles):
    col = lib.METRICS.index
    served, entered = trace[:, col("served")].sum(), trace[:, col("entered")].sum()
    income, minutes = trace[:, col("driver_total_income")].sum(), trace[:, col("online_vehicles")].sum()
    return {"served": served, "entered": entered, "served_ratio": served / entered,
            "wait": trace[:, col("wait_minutes")].sum() / served, "pickup": trace[:, col("pickup_minutes")].sum() / served,
            "left": trace[:, col("left")].sum(), "online_rate": minutes / (len(trace) * vehicles),
            "hourly_wage": income / minutes * 60.0, "profit": trace[:, col("net_profit")].sum()}


def run_cell(job):
    key, gamma, share, seed = job
    scenario = _CTX["scenarios"][key]
    market = lib.market_namespace({}, dict(MARKET, passenger_wait_sensitivity=gamma))
    policy = lib.constant_policy(scenario, 1.0, share)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=_CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    return job, cell_metrics(trace, len(scenario.vp))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=480)
    parser.add_argument("--profile_scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--ratio_range", nargs=2, type=float, default=[0.85, 0.93])
    parser.add_argument("--demand", nargs="+", type=float, default=DEMAND)
    parser.add_argument("--fleet", nargs="+", type=float, default=FLEET)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "grid_small_dynamic" / "4" / "4"
    cache = SCENARIOS / "_scan"
    cache.mkdir(parents=True, exist_ok=True)
    (SCENARIOS / ".gitignore").write_text("*\n", encoding="utf-8")
    scenarios = {}
    for d in args.demand:
        for f in args.fleet:
            key = "d{:g}_f{:g}".format(d, f)
            folder = cache / key
            folder.mkdir(exist_ok=True)
            # fractional fleets: build with the next integer multiplier, then keep the first round(f * n) drivers
            arrays = build(source, int(np.ceil(f)), args.profile_scale, d, 0.05, 0)
            arrays["vp"] = arrays["vp"][: int(round(f * len(np.load(source / "vp.npy"))))]
            for name, value in arrays.items():
                np.save(folder / (name + ".npy"), value)
            scenarios[key] = lib.Scenario(folder)
    _CTX.update(scenarios=scenarios, warmup=args.warmup)
    seeds = [args.seed + r for r in range(args.replicates)]
    jobs = [(k, g, s, sd) for k in scenarios for g in GAMMA for s in SHARES for sd in seeds]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_cell, jobs, chunksize=1)
    by = {}
    for (k, g, s, sd), m in results:
        by.setdefault((k, g), {}).setdefault(s, {})[sd] = m
    table = []
    for (k, g), per_s in sorted(by.items()):
        row = {"scenario": k, "gamma": g, "vehicles": len(scenarios[k].vp)}
        for s in SHARES:
            reps = [per_s[s][sd] for sd in seeds]
            row["s{:g}".format(s)] = {name: float(np.mean([r[name] for r in reps])) for name in reps[0]}
        lo, hi = [np.array([per_s[s][sd]["served"] for sd in seeds]) for s in (0.6, 0.9)]
        arc = ((hi - lo) / ((hi + lo) / 2)) / ((0.9 - 0.6) / 0.75)
        mean, low, high = lib.confidence_interval(arc)
        row["elasticity_served_wrt_s"] = {"mean": mean, "ci95": [low, high]}
        ratio = row["s0.75"]["served_ratio"]
        row["qualifies"] = bool(args.ratio_range[0] <= ratio <= args.ratio_range[1] and low > 0)
        table.append(row)
    (out / "supply_scan.json").write_text(json.dumps({"market": MARKET, "replicates": args.replicates, "table": table}, indent=1), encoding="utf-8")
    print("{:>10} {:>5} {:>4} | {:>7} {:>6} {:>6} {:>7} {:>6} {:>7} | {:>6} {:>17} {}".format(
        "scenario", "gamma", "veh", "ratio", "wait", "pickup", "left", "online", "wage", "e_s", "95% CI", "ok"))
    for r in table:
        a = r["s0.75"]
        e = r["elasticity_served_wrt_s"]
        print("{:>10} {:>5g} {:>4} | {:>7.3f} {:>6.2f} {:>6.2f} {:>7.0f} {:>6.3f} {:>7.1f} | {:>6.3f} [{:>6.3f},{:>6.3f}] {}".format(
            r["scenario"], r["gamma"], r["vehicles"], a["served_ratio"], a["wait"], a["pickup"], a["left"], a["online_rate"],
            a["hourly_wage"], e["mean"], *e["ci95"], "YES" if r["qualifies"] else ""))
    return table


if __name__ == "__main__":
    main()
