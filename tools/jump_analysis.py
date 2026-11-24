"""What changes across a jump of the Lagrangian frontier?

For every pair of adjacent lambda points whose mean G jumps by more than --jump (default 0.2) in policy class (a) or (b),
both solutions are re-simulated on the 20 evaluation days and compared: online minutes, served orders, per-block shares,
and the online-rate curve by hour.  Verdict "compression" = the high-G side has clearly fewer online minutes than the
low-G side (higher metric reached by shrinking the online fleet), "expansion" = clearly more, otherwise "no change".
usage: jump_analysis.py --input frontier.json --output out.json [--png out.png]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import static_decoupling as dec  # noqa: E402
from tools import frontier_decoupling as fr  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402

THRESHOLD = 0.03  # relative change of online minutes that counts as compression / expansion


def curve_job(job):
    canon, seed = job
    scenario, market = dec._CTX["scenario"], dec._CTX["market"]
    policy = dec.make_policy(canon)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=dec._CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    col = lib.METRICS.index
    online = trace[:, col("online_vehicles")]
    fleet = float(len(scenario.vp))
    return {"rate_by_hour": [float(online[h * 60:(h + 1) * 60].mean() / fleet) for h in range(24)],
            "online_minutes": float(online.sum()), "served": float(trace[:, col("served")].sum()),
            "profit": float(trace[:, col("net_profit")].sum()), "consumer_surplus": float(trace[:, col("consumer_surplus")].sum())}


def side(pool, entry, kind, seeds):
    canon = np.r_[np.array(entry[kind]["prices"]), np.array(entry[kind]["shares"])]
    out = pool.map(curve_job, [(canon, s) for s in seeds])
    mean = lambda k: float(np.mean([o[k] for o in out]))
    return {"lambda": entry["lambda"], "g_rel": float(np.mean(entry[kind]["days"]["g_rel"])), "prices": entry[kind]["prices"],
            "shares_by_block": np.array(entry[kind]["shares"]).reshape(4, 2).tolist() if len(entry[kind]["shares"]) == 8
            else entry[kind]["shares"], "online_minutes": mean("online_minutes"), "served": mean("served"), "profit": mean("profit"),
            "consumer_surplus": mean("consumer_surplus"),
            "rate_by_hour": np.mean([o["rate_by_hour"] for o in out], axis=0).tolist()}


def verdict(lo, hi):
    change = hi["online_minutes"] / lo["online_minutes"] - 1.0
    return {"online_minutes_change": change, "served_change": hi["served"] / lo["served"] - 1.0,
            "verdict": "compression" if change < -THRESHOLD else "expansion" if change > THRESHOLD else "no change"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--png", default=None)
    ap.add_argument("--jump", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=480)
    args = ap.parse_args(argv)
    fr.setup(args)
    doc = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pts = sorted(doc["points"], key=lambda p: p["lambda"])
    seeds = doc["final_seeds"]
    result = {"input": args.input, "jumps": []}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for kind in ("a", "b"):
            g = [float(np.mean(p[kind]["days"]["g_rel"])) for p in pts]
            for i in range(len(pts) - 1):
                if g[i + 1] - g[i] > args.jump:
                    lo, hi = side(pool, pts[i], kind, seeds), side(pool, pts[i + 1], kind, seeds)
                    result["jumps"].append(dict(kind=kind, low=lo, high=hi, **verdict(lo, hi)))
                    print("{} lambda {:.0f}->{:.0f}: G {:.3f}->{:.3f}, online minutes {:+.1%}, served {:+.1%}: {}".format(
                        kind, lo["lambda"], hi["lambda"], lo["g_rel"], hi["g_rel"], result["jumps"][-1]["online_minutes_change"],
                        result["jumps"][-1]["served_change"], result["jumps"][-1]["verdict"]), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.png and result["jumps"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(result["jumps"])
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.6), squeeze=False)
        for ax, j in zip(axes[0], result["jumps"]):
            for key, color in (("low", "#1b6ca8"), ("high", "#d1495b")):
                ax.plot(range(24), j[key]["rate_by_hour"], color=color,
                        label="lambda {:.0f}, G {:.2f}".format(j[key]["lambda"], j[key]["g_rel"]))
            ax.set(title="({}) online rate by hour: {}".format(j["kind"], j["verdict"]), xlabel="hour", ylabel="online rate")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
    return result


if __name__ == "__main__":
    main()
