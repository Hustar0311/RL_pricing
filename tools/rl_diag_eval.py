"""RL diagnostic evaluation: TD3 checkpoints vs the best constant uniform price under the static_sweep protocol.

Same protocol as fixed_share_rl_eval (1440-minute warm-up, 5 replicates, seeds 2000-2004).  ``kind=joint`` loads
joint-runner checkpoints (fixed_share / fixed_share_global ...) from ``joint_checkpoint_<epoch>.pt``; ``kind=legacy``
loads the price_only checkpoints (``actor_<epoch>.pkl``) of main.py's legacy path (driver share = delta, q = p).
usage: rl_diag_eval.py --scenario S --market k=v,... --share s --kind joint|legacy --dir D --epochs 5 10 ... --output O
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
from tools import static_sweep as sw  # noqa: E402
from tools.legacy_policy import load_legacy_policy  # noqa: E402

PRICE_GRID = [round(0.5 + 0.1 * i, 2) for i in range(19)]


def load(kind, directory, epoch, scenario):
    if kind == "legacy":
        return load_legacy_policy(directory, epoch, scenario)
    return lib.load_policy(directory, epoch, scenario)[0]


def td3_job(job):
    kind, directory, epoch, seed = job
    scenario, market = dec._CTX["scenario"], dec._CTX["market"]
    policy = load(kind, directory, epoch, scenario)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=dec._CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace, actions = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held, record_actions=True)
    summary = sw.summarize_trace(trace)
    summary["mean_price"], summary["mean_share"] = float(actions[..., 0].mean()), float(actions[..., 2].mean())
    summary["price_sd_zone"] = float(actions[..., 0].std(axis=1).mean())
    return summary


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def exists(kind, directory, epoch):
    name = "joint_checkpoint_{}.pt" if kind == "joint" else "actor_{}.pkl"
    return (Path(directory) / name.format(epoch)).exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--share", type=float, required=True)
    ap.add_argument("--kind", choices=["joint", "legacy"], default="joint")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--epochs", nargs="+", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--price_max", type=float, default=2.3, help="upper end of the constant-price grid (step 0.1 from 0.5)")
    args = ap.parse_args(argv)
    args.warmup = 1440
    fr.setup(args)
    log = lambda m: print(m, flush=True)
    seeds = [2000 + i for i in range(args.replicates)]
    prices_s = lambda p: np.r_[np.full(4, p), np.full(8, args.share)]
    result = {"scenario": args.scenario, "market": args.market, "share": args.share, "kind": args.kind}
    with mp.get_context("fork").Pool(args.workers) as pool:
        grid = [round(0.5 + 0.1 * i, 2) for i in range(int(round((args.price_max - 0.5) / 0.1)) + 1)]
        const = {p: pool.map(dec.run_job, [(prices_s(p), s) for s in seeds]) for p in grid}
        best = max(const, key=lambda p: mean(const[p], "profit"))
        for d in (-0.075, -0.05, -0.025, 0.025, 0.05, 0.075):
            p = round(best + d, 3)
            const[p] = pool.map(dec.run_job, [(prices_s(p), s) for s in seeds])
        best = max(const, key=lambda p: mean(const[p], "profit"))
        result["constant"] = {"best_price": best, "best_profit": mean(const[best], "profit"),
                              "profit_by_price": {str(p): mean(const[p], "profit") for p in sorted(const)}}
        log("best constant {}".format(result["constant"]["best_price"]))
        rows_out = []
        for epoch in args.epochs:
            if not exists(args.kind, args.dir, epoch):
                continue
            rows = pool.map(td3_job, [(args.kind, args.dir, epoch, s) for s in seeds])
            entry = {"epoch": epoch, "profit": mean(rows, "profit"), "mean_price": mean(rows, "mean_price"),
                     "mean_share": mean(rows, "mean_share"), "price_sd_zone": mean(rows, "price_sd_zone"),
                     "consumer_surplus": mean(rows, "consumer_surplus"), "hourly_wage": mean(rows, "hourly_wage"),
                     "served": mean(rows, "served"), "profit_std": float(np.std([r["profit"] for r in rows], ddof=1)),
                     "gap_to_best_constant_pct": 100 * (mean(rows, "profit") / result["constant"]["best_profit"] - 1)}
            rows_out.append(entry)
            log("epoch {}: {}".format(epoch, entry))
    result["checkpoints"] = rows_out
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
