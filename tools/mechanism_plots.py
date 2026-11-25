"""Mechanism figures: where do the (a) and second-class ((b4) or (b)) solutions differ over the day?

For each target of a --repeats JSON the repeat whose Delta = profit(second) - profit(a) is the median over repeats is
selected (lower median for an even count); both solutions are re-simulated on the evaluation days (seeds 5000-5019,
warm-up 480) and, per six-hour block, the mean over days of: posted price, driver share (mean over zone groups),
online vehicle-minutes, served orders, entering demand (passengers who accept the posted price) and potential demand
(the exogenous request counts), mean wait per served order and number of timed-out leaving passengers are drawn side by side.  One figure per JSON (= per constraint form).
usage: mechanism_plots.py --scenario S --market k=v,... --repeats_json J --label NAME --output_dir D
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import frontier_decoupling as fr  # noqa: E402
from tools import static_decoupling as dec  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402

BLOCK_LABELS = ["0-6h", "6-12h", "12-18h", "18-24h"]


def day_trace(job):
    canon, seed = job
    scenario, market = dec._CTX["scenario"], dec._CTX["market"]
    policy = dec.make_policy(canon)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=dec._CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    return lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)


def block_means(traces, name):
    col = lib.METRICS.index(name)
    out = np.array([[t[b * dec.BLOCK_MINUTES:(b + 1) * dec.BLOCK_MINUTES, col].sum() for b in range(dec.BLOCKS)] for t in traces])
    return out.mean(axis=0)


def median_rep(reps):
    deltas = [np.mean(r["b"]["days"]["profit"]) - np.mean(r["a"]["days"]["profit"]) for r in reps]
    order = np.argsort(deltas)
    return int(order[(len(order) - 1) // 2]), deltas


def summarize(canon, seeds, pool, scenario):
    traces = pool.map(day_trace, [(canon, s) for s in seeds])
    online = block_means(traces, "online_vehicles")
    served, entered = block_means(traces, "served"), block_means(traces, "entered")
    potential = np.array([scenario.dd_test[b * dec.BLOCK_MINUTES:(b + 1) * dec.BLOCK_MINUTES].sum() for b in range(dec.BLOCKS)])
    prices, shares = np.asarray(canon[:dec.BLOCKS]), np.asarray(canon[dec.BLOCKS:]).reshape(dec.BLOCKS, 2).mean(axis=1)
    profit = block_means(traces, "net_profit")
    wait = block_means(traces, "wait_minutes") / np.maximum(served, 1e-9)  # mean wait per served order (minutes)
    left = block_means(traces, "left")  # passengers who time out and leave
    return {"wait_per_order": wait.tolist(), "left": left.tolist(), "price": prices.tolist(), "share": shares.tolist(), "online_minutes": online.tolist(), "served": served.tolist(),
            "entered": entered.tolist(), "potential_demand": potential.tolist(), "profit": profit.tolist(),
            "profit_total": float(profit.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--repeats_json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    args.warmup = 480
    fr.setup(args)
    scenario = dec._CTX["scenario"]
    doc = json.loads(Path(args.repeats_json).read_text(encoding="utf-8"))
    second = doc.get("second_kind", "b")
    seeds = doc["final_seeds"]
    result = {"label": args.label, "second_kind": second, "targets": []}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for item in doc["repeats"]:
            index, deltas = median_rep(item["reps"])
            rep = item["reps"][index]
            entry = {"target": item["target"], "median_repeat": index, "deltas": deltas}
            for key in ("a", "b"):
                canon = np.r_[rep[key]["prices"], rep[key]["shares"]]
                entry[key] = summarize(canon, seeds, pool, scenario)
            result["targets"].append(entry)
            print("target {}: repeat {} delta {:+.0f}".format(item["target"], index, deltas[index]), flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / (args.label + ".json")).write_text(json.dumps(result, indent=1), encoding="utf-8")
    draw(result, out / (args.label + ".png"), second)
    return result


def draw(result, path, second):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("price", "posted price"), ("share", "driver share"), ("online_minutes", "online vehicle-minutes"),
            ("served", "served orders"), ("entered", "entering demand"), ("wait_per_order", "wait per order (min)"),
            ("left", "passengers leaving (timeout)")]
    n = len(result["targets"])
    fig, axes = plt.subplots(len(rows), n, figsize=(5.2 * n, 2.2 * len(rows)), sharex=True, squeeze=False)
    x = np.arange(dec.BLOCKS)
    for j, entry in enumerate(result["targets"]):
        for i, (key, title) in enumerate(rows):
            ax = axes[i][j]
            ax.bar(x - 0.2, entry["a"][key], 0.4, label="(a) p + s", color="#4C78A8")
            ax.bar(x + 0.2, entry["b"][key], 0.4, label="({}) median repeat".format(second), color="#F58518")
            if key == "entered":
                ax.plot(x, entry["a"]["potential_demand"], "k--", lw=1, label="potential demand")
            ax.set_ylabel(title, fontsize=8)
            if i == 0:
                ax.set_title("{} W={} ($\\Delta$ = {:+.0f})".format(result["label"], entry["target"], entry["deltas"][entry["median_repeat"]]), fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
        axes[-1][j].set_xticks(x)
        axes[-1][j].set_xticklabels(BLOCK_LABELS)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
