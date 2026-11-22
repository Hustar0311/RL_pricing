"""Summarise share-mode evaluation runs and decompose Delta = joint_share - best benchmark.

Reads test_log_1..5.csv of each run (the five evaluation repeats share demand and random seeds, so repeats are
paired across runs).  Delta is the platform-profit gap between joint_share and the best of learned_share and the
fixed shares; transfer = (take-rate change) x (passenger revenue, midpoint of the two runs), efficiency = the rest.
Single-seed output: a pipeline check, not a research conclusion.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from welfare import aggregate_welfare  # noqa: E402


def run_metrics(folder):
    rows = []
    for path in sorted(glob.glob(str(Path(folder) / "test_log_*.csv"))):
        frame = pd.read_csv(path)
        welfare = aggregate_welfare(frame)
        rows.append({
            "profit": float(frame["reward"].sum()), "revenue": float(frame["profit"].sum()),
            "take_rate": welfare["platform_take_rate"], "driver_income": welfare["driver_total_income"],
            "hourly_wage": welfare["driver_hourly_wage"], "consumer_surplus": welfare["consumer_surplus"],
            "served": float(frame["served"].sum()), "online_vehicles": float(frame["act_veh"].mean()),
            "mean_price": float(frame["price_mean"].mean()),
            "mean_share": float(frame["share_mean"].mean()), "share_zone_std": float(frame["share_std"].mean()),
            "online_rate": float(frame["driver_online_rate"].mean()) if "driver_online_rate" in frame else float("nan"),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_glob", required=True, help="glob with {label} for the test folders")
    parser.add_argument("--fixed", nargs="+", default=["fixed0.65", "fixed0.75", "fixed0.85"])
    parser.add_argument("--learned", default="learned")
    parser.add_argument("--joint", default="jointshare")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    labels = args.fixed + [args.learned, args.joint]
    runs = {}
    for label in labels:
        hits = sorted(glob.glob(args.results_glob.format(label=label)))
        if len(hits) != 1:
            raise SystemExit("expected exactly one folder for {} got {}".format(label, hits))
        runs[label] = run_metrics(hits[0])
    columns = ["profit", "take_rate", "driver_income", "hourly_wage", "consumer_surplus", "served", "online_vehicles",
               "mean_price", "mean_share", "share_zone_std"]
    table = pd.DataFrame({label: {c: runs[label][c].mean() for c in columns} for label in labels}).T
    spread = pd.DataFrame({label: {c: runs[label][c].std(ddof=1) for c in columns} for label in labels}).T
    print(table.round(3).to_string())
    print("\n(std over the 5 evaluation repeats)\n" + spread.round(3).to_string())
    benchmarks = args.fixed + [args.learned]
    best = max(benchmarks, key=lambda l: runs[l]["profit"].mean())
    joint, base = runs[args.joint], runs[best]
    delta = joint["profit"] - base["profit"]
    transfer = (joint["take_rate"] - base["take_rate"]) * (joint["revenue"] + base["revenue"]) / 2
    efficiency = delta - transfer

    def ci(series):
        n = len(series)
        se = series.std(ddof=1) / np.sqrt(n)
        return float(series.mean()), float(se)

    result = {"best_benchmark": best, "benchmarks_profit": {l: float(runs[l]["profit"].mean()) for l in benchmarks},
              "delta": ci(delta), "transfer": ci(transfer), "efficiency": ci(efficiency),
              "note": "single seed (47); standard errors reflect only the 5 evaluation repeats"}
    print("\nbest benchmark: {}".format(best))
    for key in ["delta", "transfer", "efficiency"]:
        print("{:>10}: {:+.2f} (se {:.2f})".format(key, *result[key]))
    if args.output:
        Path(args.output).write_text(json.dumps({"table": table.to_dict("index"), "spread": spread.to_dict("index"),
                                                 "decomposition": result}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
