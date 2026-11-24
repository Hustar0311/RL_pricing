"""Fair RL test: price-only TD3 at a pinned driver share against the best static prices at the same share.

  * TD3 checkpoints (fixed_share, loaded with the fixed load_policy) are evaluated in the static_sweep protocol
    (1440-minute warm-up, 5 replicates, seeds 2000-2004): profit, mean price / share actually used.
  * best constant (uniform) price at that share: price sweep in the same protocol.
  * value of dynamic pricing, upper bound for a time-of-day price policy: (f) uniform price vs (g) four block prices at
    the same share, searched on the optimisation days (480-minute warm-up, seeds 4000-4009) and evaluated on 20 new days
    (seeds 5000-5019) - the fixed-share analogue of (a') - (a); the two solutions are also re-evaluated in the sweep protocol.
usage: fixed_share_rl_eval.py --scenario S --market k=v,... --share 0.4 --run label=dir ... --epochs 10 20 30 --output out.json
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
from tools.epsilon_decoupling import boot_mean  # noqa: E402

PRICE_GRID = [round(0.8 + 0.1 * i, 2) for i in range(13)]


def td3_job(job):
    directory, epoch, seed = job
    scenario, market = dec._CTX["scenario"], dec._CTX["market"]
    policy = lib.load_policy(directory, epoch, scenario)[0]
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=dec._CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace, actions = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held, record_actions=True)
    summary = sw.summarize_trace(trace)
    summary["mean_price"], summary["mean_share"] = float(actions[..., 0].mean()), float(actions[..., 2].mean())
    return summary


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--share", type=float, required=True)
    ap.add_argument("--run", nargs="+", required=True, help="label=checkpoint_dir")
    ap.add_argument("--epochs", nargs="+", type=int, required=True, help="checkpoint epochs to evaluate (missing ones are skipped)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sweep_seed", type=int, default=2000)
    ap.add_argument("--sweep_replicates", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    dec.FIXED_SHARE = args.share
    args.warmup = 480
    log = lambda m: print(m, flush=True)
    result = {"scenario": args.scenario, "market": args.market, "share": args.share}
    opt_seeds, final_seeds = [4000 + i for i in range(10)], [5000 + i for i in range(20)]
    sweep_seeds = [args.sweep_seed + i for i in range(args.sweep_replicates)]
    prices_s = lambda p: (np.full(4, p), np.full(8, args.share))

    # phase 1: dynamic-pricing value at the pinned share (optimisation protocol)
    fr.setup(args)
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = dec.Evaluator(pool)
        base_rows = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], final_seeds)[0]
        con = fr.Lagrangian("hourly", base_rows, 0.0)  # lambda = 0: profit only
        rng = np.random.default_rng(777)
        xf, ff, _ = dec.search(ev, con, "f", [prices_s(1.0)], rng, opt_seeds, 1.0, args.restarts, log)
        xg, fg, _ = dec.search(ev, con, "g", [dec.canonical("f", xf), prices_s(1.0)], rng, opt_seeds, 1.0, args.restarts, log)
        rows_f, rows_g = (fr.point_rows(ev, con, k, x, final_seeds) for k, x in (("f", xf), ("g", xg)))
    pf, pg = (np.array([r["profit"] for r in rows]) for rows in (rows_f, rows_g))
    result["dynamic_value"] = {"uniform_price": float(dec.canonical("f", xf)[0][0]), "block_prices": dec.canonical("g", xg)[0].tolist(),
                               "profit_uniform": float(pf.mean()), "profit_blocks": float(pg.mean()),
                               "gain": boot_mean(pg - pf, args.bootstrap), "gain_pct": float(100 * (pg.mean() / pf.mean() - 1))}
    log("dynamic value (g)-(f): {}".format(result["dynamic_value"]))

    # phase 2: TD3 and static prices in the static_sweep protocol (1440-minute warm-up)
    args.warmup = 1440
    fr.setup(args)
    results = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        const = {}
        for p in PRICE_GRID:
            canon = np.r_[prices_s(p)[0], prices_s(p)[1]]
            const[p] = pool.map(dec.run_job, [(canon, s) for s in sweep_seeds])
        best = max(const, key=lambda p: mean(const[p], "profit"))
        fine = {}
        for p in [round(best + d, 3) for d in (-0.075, -0.05, -0.025, 0.025, 0.05, 0.075) if best + d >= 0.8]:
            fine[p] = pool.map(dec.run_job, [(np.r_[prices_s(p)[0], prices_s(p)[1]], s) for s in sweep_seeds])
        const.update(fine)
        best = max(const, key=lambda p: mean(const[p], "profit"))
        result["constant"] = {"best_price": best, "best_profit": mean(const[best], "profit"),
                              "profit_by_price": {str(p): mean(const[p], "profit") for p in sorted(const)}}
        for name, x, kind in (("uniform_price_static", xf, "f"), ("block_prices_static", xg, "g")):
            rows = pool.map(dec.run_job, [(np.r_[dec.canonical(kind, x)[0], dec.canonical(kind, x)[1]], s) for s in sweep_seeds])
            result[name] = {"profit": mean(rows, "profit"), "prices": dec.canonical(kind, x)[0].tolist()}
        for item in args.run:
            label, _, directory = item.partition("=")
            results[label] = []
            for epoch in args.epochs:
                if not (Path(directory) / "joint_checkpoint_{}.pt".format(epoch)).exists():
                    continue
                rows = pool.map(td3_job, [(directory, epoch, s) for s in sweep_seeds])
                entry = {"epoch": epoch, "profit": mean(rows, "profit"), "mean_price": mean(rows, "mean_price"),
                         "mean_share": mean(rows, "mean_share"), "consumer_surplus": mean(rows, "consumer_surplus"),
                         "hourly_wage": mean(rows, "hourly_wage"), "served": mean(rows, "served"),
                         "profit_std": float(np.std([r["profit"] for r in rows], ddof=1)),
                         "gap_to_best_constant_pct": 100 * (mean(rows, "profit") / result["constant"]["best_profit"] - 1)}
                results[label].append(entry)
                log("{} epoch {}: {}".format(label, epoch, entry))
    result["td3"] = results
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
