"""TD3 checkpoint vs the best constant price at the same driver share (re-evaluation with the fixed load_policy).

For each checkpoint: five replicates under the static_sweep protocol (warm-up under the policy, one test day); mean
price/share actually used (recorded actions); profit; best constant price at the policy's mean share, by (i) linear
interpolation of the stored sweep grid (the earlier v1 diagnostic) and (ii) a direct sweep over the price grid at that
share with the same seeds.  usage: td3_vs_constant.py --scenario S --grid_json G --output O --run label=dir ...
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402
from tools import static_sweep as sw  # noqa: E402


def policy_actions(job):
    scenario, market = sw._CONTEXT["scenario"], sw._CONTEXT["market"]
    policy = sw._CONTEXT["policy_factory"]()
    state, held, t0 = lib.build_start_state(scenario, policy, market, 0, job, warmup_minutes=sw._CONTEXT["warmup"])
    env = lib.make_environment(scenario, market, job, policy.frequency)
    env.restore_state(state)
    trace, actions = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, job, None, held, record_actions=True)
    return sw.summarize_trace(trace), actions


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--grid_json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--run", nargs="+", required=True, help="label=checkpoint_dir")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    scenario = lib.Scenario(args.scenario)
    market = lib.market_namespace({}, dict(sw.MARKET))
    sw._CONTEXT.update(scenario=scenario, market=market, warmup=1440)
    seeds = [args.seed + r for r in range(args.replicates)]
    grid_doc = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
    prices, shares, grid = grid_doc["prices"], grid_doc["shares"], grid_doc["grid"]
    best = [max(grid["{:g}|{:g}".format(p, s)]["profit"]["mean"] for p in prices) for s in shares]
    out = {}
    for item in args.run:
        label, _, directory = item.partition("=")
        sw._CONTEXT["policy_factory"] = lambda d=directory: lib.load_policy(d, args.epoch, scenario)[0]
        with mp.get_context("fork").Pool(min(args.workers, len(seeds))) as pool:  # fork after the factory is set
            res = pool.map(policy_actions, seeds)
            acts_mean_s = float(np.concatenate([r[1].reshape(-1, 3) for r in res])[:, 2].mean())
            direct = pool.map(sw.run_point, [("grid", p, acts_mean_s, seed) for p in prices for seed in seeds])
        profit = float(np.mean([r[0]["profit"] for r in res]))
        acts = np.concatenate([r[1].reshape(-1, 3) for r in res])
        mean_p, mean_s = float(acts[:, 0].mean()), float(acts[:, 2].mean())
        by_p = {p: float(np.mean([m["profit"] for (_, pp, _, _), m in direct if pp == p])) for p in prices}
        bp = max(by_p, key=by_p.get)
        interp = float(np.interp(mean_s, shares, best))
        out[label] = {"profit": profit, "mean_price": mean_p, "mean_share": mean_s, "best_constant_interpolated": interp,
                      "gap_interpolated_pct": 100 * (profit / interp - 1), "best_constant_direct": by_p[bp], "best_constant_price": bp,
                      "gap_direct_pct": 100 * (profit / by_p[bp] - 1), "constant_profit_by_price": by_p}
        print(label, json.dumps({k: v for k, v in out[label].items() if k != "constant_profit_by_price"}), flush=True)
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
