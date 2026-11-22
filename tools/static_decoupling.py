"""Static (no RL) decoupling benchmark under a driver hourly-wage floor.

Two parameterised policies are optimised on the same scenario with a penalty for violating the floor W-bar:
  (a) fixed share:  one uniform price p and one global driver share s            (2 parameters)
  (b) decoupled:    one uniform price p and 8 shares, one per (time block x zone group): 4 six-hour blocks
                    x {low-demand zones, high-demand zones} (groups split at the median origin demand)  (9 parameters)
Search: a (mu/mu_w, lambda) evolution strategy with step-size adaptation followed by coordinate pattern-search
refinement; every candidate is scored on --opt_replicates (5) common-random-number days.  The final comparison uses
--final_replicates (20) fresh days.  Delta_static = (b) - (a) is reported with a paired 95% t-interval and split into
  allocation = -(change in total driver income)   and   efficiency = the rest (revenue less non-driver costs; detail rows:
  served orders, wait, pickup, empty minutes, expired passengers, online minutes).
W-bar defaults to the driver hourly wage at p=1, s=0.75 and is then scaled by --floor_factors.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402

MARKET = dict(use_passenger_entry_br=True, use_driver_online_br=True, passenger_calibration="anchor",
              driver_decision_interval=30, passenger_wait_sensitivity=0.5)
P_BOUNDS, S_BOUNDS = (0.8, 2.0), (0.1, 0.95)
BLOCKS, BLOCK_MINUTES = 4, 360
PENALTY = 3000.0  # profit units per $/h of wage shortfall (>> the profit value of an extra $/h, ~700)
KINDS = {"fixed": 2, "decoupled": 9}
_CTX = {}


def make_policy(kind, theta):
    base = copy.copy(_CTX["base"])
    group = _CTX["group"]
    if kind == "fixed":
        price, share = float(theta[0]), float(theta[1])
        base.constant = (price, share)
    else:
        table = np.asarray(theta[1:], dtype=float).reshape(BLOCKS, 2)
        price = float(theta[0])
        base.constant = lambda t, table=table, price=price: (price, table[min(int(t) // BLOCK_MINUTES, BLOCKS - 1)][group])
    return base


def run_job(job):
    kind, theta, seed = job
    scenario, market = _CTX["scenario"], _CTX["market"]
    policy = make_policy(kind, theta)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=_CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    col = lib.METRICS.index
    served, minutes = trace[:, col("served")].sum(), trace[:, col("online_vehicles")].sum()
    income = trace[:, col("driver_total_income")].sum()
    return {"profit": trace[:, col("net_profit")].sum(), "revenue": trace[:, col("revenue")].sum(), "income": income,
            "wage": income / minutes * 60.0, "consumer_surplus": trace[:, col("consumer_surplus")].sum(), "served": served,
            "entered": trace[:, col("entered")].sum(), "wait": trace[:, col("wait_minutes")].sum() / served,
            "pickup": trace[:, col("pickup_minutes")].sum() / served, "empty_minutes": trace[:, col("empty_minutes")].sum(),
            "left": trace[:, col("left")].sum(), "online_minutes": minutes}


class Evaluator:
    def __init__(self, pool):
        self.pool = pool

    def run(self, kind, thetas, seeds):
        jobs = [(kind, tuple(float(x) for x in th), sd) for th in thetas for sd in seeds]
        out = self.pool.map(run_job, jobs, chunksize=1)
        n = len(seeds)
        return [out[i * n:(i + 1) * n] for i in range(len(thetas))]

    def objective(self, kind, thetas, seeds, floor):
        scores = []
        for rows in self.run(kind, thetas, seeds):
            profit = np.mean([r["profit"] for r in rows])
            wage = np.mean([r["wage"] for r in rows])
            scores.append(profit - PENALTY * max(0.0, floor - wage))
        return np.array(scores)


def bounds(kind):
    d = KINDS[kind]
    lo = np.array([P_BOUNDS[0]] + [S_BOUNDS[0]] * (d - 1))
    hi = np.array([P_BOUNDS[1]] + [S_BOUNDS[1]] * (d - 1))
    scales = np.array([0.15] + [0.08] * (d - 1))
    return lo, hi, scales


def optimize(ev, kind, x0, floor, seeds, gens, lam, rng, log):
    lo, hi, scales = bounds(kind)
    mean, sigma = np.clip(np.asarray(x0, dtype=float), lo, hi), 1.0
    best_x, best_f = mean.copy(), -np.inf
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    for g in range(gens):
        X = np.clip(mean + sigma * scales * rng.standard_normal((lam, len(mean))), lo, hi)
        X[0] = mean
        F = ev.objective(kind, X, seeds, floor)
        order = np.argsort(-F)
        improved = F[order[0]] > best_f + 1e-9
        if improved:
            best_f, best_x = float(F[order[0]]), X[order[0]].copy()
        mean = weights @ X[order[:mu]]
        sigma = float(np.clip(sigma * (1.15 if improved else 0.8), 0.05, 1.5))
        log("  [{}] gen {:>2}/{} best={:.0f} sigma={:.2f}".format(kind, g + 1, gens, best_f, sigma))
    # coordinate pattern search around the best point
    step = scales * 0.5
    for _ in range(4):
        while True:
            neighbours = []
            for i in range(len(best_x)):
                for sign in (-1.0, 1.0):
                    cand = best_x.copy()
                    cand[i] = np.clip(cand[i] + sign * step[i], lo[i], hi[i])
                    neighbours.append(cand)
            F = ev.objective(kind, neighbours, seeds, floor)
            if F.max() > best_f + 1e-9:
                best_f, best_x = float(F.max()), neighbours[int(F.argmax())]
            else:
                break
        step = step / 2
        log("  [{}] refine step {} best={:.0f}".format(kind, np.round(step[:2], 3), best_f))
    return best_x, best_f


def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def paired_delta(rows_b, rows_a, kappa=None):
    """Per-seed (b - a) with mean and 95% CI; allocation = -d(income), efficiency = d(profit) + d(income)."""
    keys = list(rows_a[0])
    diff = {k: np.array([b[k] - a[k] for a, b in zip(rows_a, rows_b)]) for k in keys}
    diff["allocation"] = -diff["income"]
    diff["efficiency"] = diff["profit"] + diff["income"]
    if kappa is not None:
        # wage-neutral version: price the per-seed wage gap at kappa profit units per $/h
        diff["profit_wage_adjusted"] = diff["profit"] + kappa * diff["wage"]
    return {k: dict(zip(["mean", "lo", "hi"], lib.confidence_interval(v))) for k, v in diff.items()}


def shift_shares(kind, theta, offset):
    theta = np.array(theta, dtype=float)
    theta[1:] = np.clip(theta[1:] + offset, S_BOUNDS[0], S_BOUNDS[1])
    return theta


def project_to_floor(ev, kind, theta, floor, seeds, tolerance=0.05):
    """Smallest uniform share offset in [-0.05, 0.15] whose mean wage on fresh calibration days reaches the floor.

    The penalised search is scored on 5 days, so its optimum can fall slightly short of W-bar on new days; applying the
    same projection to both policies makes the (b) - (a) comparison feasible for both and removes the profit that a
    shortfall in the wage floor would otherwise pay for.
    """
    wage = lambda off: float(np.mean([r["wage"] for r in ev.run(kind, [shift_shares(kind, theta, off)], seeds)[0]]))
    lo, hi = -0.05, 0.15
    if wage(lo) >= floor + tolerance:
        return shift_shares(kind, theta, lo), lo
    for _ in range(9):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if wage(mid) >= floor + tolerance else (mid, hi)
    return shift_shares(kind, theta, hi), hi


def finalize(args):
    """Project the stored optima onto the wage floor on fresh days and re-run the 20-day comparison."""
    doc = json.loads(Path(args.finalize_from).read_text(encoding="utf-8"))
    scenario = lib.Scenario(doc["scenario"])
    group = np.array(doc["group"])
    _CTX.update(scenario=scenario, market=lib.market_namespace({}, doc["market"]), warmup=args.warmup,
                base=lib.constant_policy(scenario, 1.0, 0.75), group=group)
    calib_seeds = [args.projection_seed + r for r in range(args.projection_replicates)]
    final_seeds = doc["final_seeds"]
    clock = time.time()
    results = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = Evaluator(pool)
        for factor, entry in doc["results"].items():
            floor, theta, rows = entry["floor"], {}, {}
            offsets = {}
            for kind in KINDS:
                theta[kind], offsets[kind] = project_to_floor(ev, kind, entry["theta"][kind], floor, calib_seeds)
                rows[kind] = ev.run(kind, [theta[kind]], final_seeds)[0]
            # marginal profit cost of one extra $/h of driver wage, from a +-0.02 uniform share shift of policy (a)
            up, down = (summarize(ev.run("fixed", [shift_shares("fixed", theta["fixed"], o)], final_seeds)[0]) for o in (0.02, -0.02))
            kappa = -(up["profit"] - down["profit"]) / (up["wage"] - down["wage"])
            results[factor] = {"floor": floor, "share_offset": offsets, "theta": {k: theta[k].tolist() for k in KINDS},
                               "final": {k: summarize(rows[k]) for k in KINDS}, "kappa_profit_per_wage": kappa,
                               "delta_static": paired_delta(rows["decoupled"], rows["fixed"], kappa)}
            d = results[factor]["delta_static"]["profit"]
            print("{:>6.0f}s x{} floor {:.2f}: offsets {} wage a={:.2f} b={:.2f} Delta {:+.0f} [{:+.0f},{:+.0f}]".format(
                time.time() - clock, factor, floor, {k: round(v, 3) for k, v in offsets.items()},
                results[factor]["final"]["fixed"]["wage"], results[factor]["final"]["decoupled"]["wage"], d["mean"], d["lo"], d["hi"]), flush=True)
    doc["projected"] = results
    Path(args.finalize_from).with_name("static_decoupling_projected.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario")
    parser.add_argument("--output_dir")
    parser.add_argument("--driver_hourly_floor", default="auto", help="W-bar, or 'auto' = wage at p=1, s=0.75")
    parser.add_argument("--floor_factors", nargs="+", type=float, default=[1.0, 0.8, 1.2])
    parser.add_argument("--opt_replicates", type=int, default=5)
    parser.add_argument("--final_replicates", type=int, default=20)
    parser.add_argument("--opt_seed", type=int, default=4000)
    parser.add_argument("--final_seed", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1440)
    parser.add_argument("--gens_fixed", type=int, default=14)
    parser.add_argument("--gens_decoupled", type=int, default=36)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--market", default="")
    parser.add_argument("--finalize_from", help="static_decoupling.json: project its optima onto the wage floor and re-evaluate")
    parser.add_argument("--projection_replicates", type=int, default=10)
    parser.add_argument("--projection_seed", type=int, default=6000)
    args = parser.parse_args(argv)
    if args.finalize_from:
        return finalize(args)
    if not args.scenario or not args.output_dir:
        parser.error("--scenario and --output_dir are required")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overrides = dict(MARKET)
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    scenario = lib.Scenario(args.scenario)
    demand = scenario.dd_train.sum(axis=(0, 2))
    group = (demand > np.median(demand)).astype(int)  # 1 = high-demand origin zones
    _CTX.update(scenario=scenario, market=lib.market_namespace({}, overrides), warmup=args.warmup,
                base=lib.constant_policy(scenario, 1.0, 0.75), group=group)
    opt_seeds = [args.opt_seed + r for r in range(args.opt_replicates)]
    final_seeds = [args.final_seed + r for r in range(args.final_replicates)]
    rng = np.random.default_rng(12345)
    clock = time.time()
    log = lambda msg: print("{:>7.0f}s {}".format(time.time() - clock, msg), flush=True)

    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = Evaluator(pool)
        reference = summarize(ev.run("fixed", [(1.0, 0.75)], final_seeds)[0])
        floor0 = reference["wage"] if args.driver_hourly_floor == "auto" else float(args.driver_hourly_floor)
        log("reference p=1 s=0.75: wage {:.2f} profit {:.0f}; W-bar = {:.2f}; zone groups {}".format(
            reference["wage"], reference["profit"], floor0, group.tolist()))
        results, previous = {}, {"fixed": np.array([1.2, 0.7]), "decoupled": None}
        for factor in args.floor_factors:
            floor = floor0 * factor
            log("== W-bar = {:.2f} (x{:g})".format(floor, factor))
            theta = {}
            theta["fixed"], _ = optimize(ev, "fixed", previous["fixed"], floor, opt_seeds, args.gens_fixed if factor == args.floor_factors[0] else 8, 12, rng, log)
            start_b = np.r_[theta["fixed"][0], np.full(8, theta["fixed"][1])] if previous["decoupled"] is None else previous["decoupled"]
            theta["decoupled"], _ = optimize(ev, "decoupled", start_b, floor, opt_seeds, args.gens_decoupled if factor == args.floor_factors[0] else 20, 24, rng, log)
            previous = {"fixed": theta["fixed"], "decoupled": theta["decoupled"]}
            rows = {k: ev.run(k, [theta[k]], final_seeds)[0] for k in KINDS}
            entry = {"floor": floor, "theta": {k: theta[k].tolist() for k in KINDS},
                     "final": {k: summarize(rows[k]) for k in KINDS},
                     "delta_static": paired_delta(rows["decoupled"], rows["fixed"])}
            results[str(factor)] = entry
            log("  fixed: p={:.3f} s={:.3f} profit={:.0f} wage={:.2f}".format(*theta["fixed"], entry["final"]["fixed"]["profit"], entry["final"]["fixed"]["wage"]))
            d = entry["delta_static"]["profit"]
            log("  decoupled: p={:.3f} shares={} profit={:.0f} wage={:.2f}".format(
                theta["decoupled"][0], np.round(theta["decoupled"][1:], 2).tolist(), entry["final"]["decoupled"]["profit"], entry["final"]["decoupled"]["wage"]))
            log("  Delta_static profit {:+.0f} [{:+.0f}, {:+.0f}]".format(d["mean"], d["lo"], d["hi"]))
            (out / "static_decoupling.json").write_text(json.dumps({"scenario": args.scenario, "market": overrides, "group": group.tolist(),
                "reference_p1_s0.75": reference, "floor0": floor0, "opt_seeds": opt_seeds, "final_seeds": final_seeds,
                "results": results}, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
