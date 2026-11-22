"""Static sweep of constant (price, driver share) policies, plus a same-protocol evaluation of a checkpoint.

For every grid point (uniform price p, global share s) the test day is simulated ``--replicates`` times with
common random numbers: a 1440-minute warm-up on the training counts under the same constant policy, then the 1440
test minutes (the protocol of ``rollout_lib.build_start_state`` + ``simulate``).  Reported per point (mean and std
over replicates): platform profit, driver hourly wage, consumer surplus, served orders, online vehicles, take rate.
A profit heat map is written and the optimum is reported as interior or on the grid boundary.

  python tools/static_sweep.py --scenario grid_small_dynamic/4/4 --label original --output_dir ../artifacts/...
  python tools/static_sweep.py --scenario S --label S --checkpoint_dir DIR --epoch 30 --grid_json grid.json
The second form evaluates the checkpoint under the identical protocol and prints its gap to the sweep optimum.
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

PRICES = [0.8, 0.9, 1.0, 1.1, 1.2]
SHARES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REFERENCE_POINT = (1.0, 0.75)  # (p, s) whose driver hourly wage defines --driver_hourly_floor auto
MARKET = dict(use_passenger_entry_br=True, use_driver_online_br=True, passenger_calibration="anchor",
              driver_decision_interval=30)
OUTPUTS = ["profit", "hourly_wage", "consumer_surplus", "served", "online_vehicles", "take_rate"]

_CONTEXT = {}


def summarize_trace(trace):
    col = lib.METRICS.index
    revenue, income = trace[:, col("revenue")].sum(), trace[:, col("driver_total_income")].sum()
    minutes = trace[:, col("online_vehicles")].sum()
    return {"profit": trace[:, col("net_profit")].sum(), "hourly_wage": income / minutes * 60.0 if minutes else np.nan,
            "consumer_surplus": trace[:, col("consumer_surplus")].sum(), "served": trace[:, col("served")].sum(),
            "online_vehicles": trace[:, col("online_vehicles")].mean(),
            "take_rate": 1.0 - income / revenue if revenue > 0 else np.nan}


def run_point(job):
    """One (policy, replicate): warm up under the policy, then simulate the whole test day."""
    kind, p, s, seed = job
    scenario, market = _CONTEXT["scenario"], _CONTEXT["market"]
    policy = lib.constant_policy(scenario, p, s) if kind == "grid" else _CONTEXT["policy_factory"]()
    state, held, t0 = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=_CONTEXT.get("warmup", 1440))
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    return job, summarize_trace(trace)


def aggregate(results):
    table = {}
    for (kind, p, s, seed), metrics in results:
        table.setdefault((p, s), []).append(metrics)
    return {"{:g}|{:g}".format(p, s): {k: {"mean": float(np.mean([m[k] for m in rows])),
                                            "std": float(np.std([m[k] for m in rows], ddof=1)) if len(rows) > 1 else 0.0}
                                       for k in OUTPUTS} for (p, s), rows in table.items()}


def optimum(grid, prices, shares, wage_floor=None):
    """Best profit point; with a driver hourly-wage floor only grid points whose mean wage >= floor qualify."""
    keys = [k for k in grid if wage_floor is None or grid[k]["hourly_wage"]["mean"] >= wage_floor]
    best = max(keys, key=lambda key: grid[key]["profit"]["mean"])
    p, s = (float(x) for x in best.split("|"))
    interior = prices[0] < p < prices[-1] and shares[0] < s < shares[-1]
    return p, s, interior, grid[best]["profit"]["mean"]


def heatmap(grid, prices, shares, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.array([[grid["{:g}|{:g}".format(p, s)]["profit"]["mean"] for p in prices] for s in shares])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    image = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(prices)), ["{:g}".format(p) for p in prices])
    ax.set_yticks(range(len(shares)), ["{:g}".format(s) for s in shares])
    ax.set_xlabel("uniform price multiplier p")
    ax.set_ylabel("global driver share s")
    best = np.unravel_index(np.argmax(z), z.shape)
    ax.plot(best[1], best[0], "r*", markersize=14)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, "{:.0f}".format(z[i, j]), ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(image, label="platform profit (test day, mean of replicates)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--market", default="", help="market overrides k=v,... (JSON values), e.g. passenger_wait_sensitivity=0.5")
    parser.add_argument("--prices", nargs="+", type=float, default=PRICES)
    parser.add_argument("--shares", nargs="+", type=float, default=SHARES)
    parser.add_argument("--warmup", type=int, default=1440)
    parser.add_argument("--driver_hourly_floor", default=None,
                        help="W-bar: a number, or 'auto' = driver hourly wage at p=1, s=0.75 in the same protocol; "
                             "only grid points with wage >= W-bar are eligible for the constrained optimum")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--grid_json", help="existing sweep result to compare the checkpoint with")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario = lib.Scenario(args.scenario)
    overrides = dict(MARKET)
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value) if value[:1] in "0123456789-.tfn[\"" else value
    market = lib.market_namespace({}, overrides)
    _CONTEXT.update(scenario=scenario, market=market, warmup=args.warmup)
    seeds = [args.seed + r for r in range(args.replicates)]

    if args.checkpoint_dir:
        _CONTEXT["policy_factory"] = lambda: lib.load_policy(args.checkpoint_dir, args.epoch, scenario)[0]
        policy_label = "checkpoint"
        with mp.get_context("fork").Pool(min(args.workers, len(seeds))) as pool:
            results = pool.map(run_point, [(policy_label, 0.0, 0.0, seed) for seed in seeds])
        rows = [m for _, m in results]
        summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1))}
                   for k in OUTPUTS}
        record = {"checkpoint_dir": args.checkpoint_dir, "epoch": args.epoch, "metrics": summary}
        if args.grid_json:
            grid_doc = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
            grid = grid_doc["grid"]
            p, s, interior, best_profit = optimum(grid, grid_doc["prices"], grid_doc["shares"])
            record["sweep_optimum"] = {"p": p, "s": s, "profit": best_profit, "interior": interior}
            record["profit_gap_policy_minus_best_constant"] = summary["profit"]["mean"] - best_profit
            # best constant price at each share, for a like-for-like share comparison
            record["best_constant_by_share"] = {
                "{:g}".format(sh): max(((pp, grid["{:g}|{:g}".format(pp, sh)]["profit"]["mean"]) for pp in grid_doc["prices"]),
                                       key=lambda x: x[1]) for sh in grid_doc["shares"]}
        print(json.dumps(record, indent=2, default=float))
        (out / "{}_checkpoint_epoch{}.json".format(args.label, args.epoch)).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        return record

    prices, shares = list(args.prices), list(args.shares)
    if args.driver_hourly_floor is not None and (REFERENCE_POINT[0] not in prices or REFERENCE_POINT[1] not in shares):
        shares = sorted(set(shares) | {REFERENCE_POINT[1]})
        prices = sorted(set(prices) | {REFERENCE_POINT[0]})
    jobs = [("grid", p, s, seed) for p in prices for s in shares for seed in seeds]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_point, jobs, chunksize=1)
    grid = aggregate(results)
    p, s, interior, best_profit = optimum(grid, prices, shares)
    doc = {"scenario": args.scenario, "market": overrides, "replicates": args.replicates, "seeds": seeds,
           "prices": prices, "shares": shares, "grid": grid, "warmup": args.warmup,
           "optimum": {"p": p, "s": s, "profit": best_profit, "interior": bool(interior)}}
    if args.driver_hourly_floor is not None:
        floor = (grid["{:g}|{:g}".format(*REFERENCE_POINT)]["hourly_wage"]["mean"]
                 if args.driver_hourly_floor == "auto" else float(args.driver_hourly_floor))
        fp, fs, finterior, fprofit = optimum(grid, prices, shares, floor)
        doc["wage_floor"] = {"floor": floor, "p": fp, "s": fs, "profit": fprofit, "interior": bool(finterior),
                             "feasible_points": sum(v["hourly_wage"]["mean"] >= floor for v in grid.values())}
        print("wage floor {:.2f}: best feasible p={:g} s={:g} profit={:.0f} interior={}".format(floor, fp, fs, fprofit, finterior))
    (out / "{}_grid.json".format(args.label)).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    heatmap(grid, prices, shares, "{}: profit, best p={:g}, s={:g}".format(args.label, p, s), out / "{}_profit_heatmap.png".format(args.label))
    print("{:>5} ".format("s\\p") + " ".join("{:>9g}".format(x) for x in prices))
    for sh in shares:
        print("{:>5g} ".format(sh) + " ".join("{:>9.0f}".format(grid["{:g}|{:g}".format(pp, sh)]["profit"]["mean"]) for pp in prices))
    print("optimum p={:g} s={:g} profit={:.0f} interior={}".format(p, s, best_profit, interior))
    return doc


if __name__ == "__main__":
    main()
