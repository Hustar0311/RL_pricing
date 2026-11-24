"""Static (no RL) decoupling benchmark: four parameterised policies under a driver-side constraint.

Policies (all use the same simulator, constant price/share schedules, common random numbers):
  (a)  one uniform price p + one global driver share s                             2 parameters
  (a') four time-block prices (6 h each) + one global share                          5 parameters
  (b)  uniform price p + eight shares (4 time blocks x {low, high}-demand zones)     9 parameters
  (b') four time-block prices + the eight shares                                    12 parameters
(b') vs (a') is the fair decoupling test (equal price flexibility); (b) vs (a) is kept as an extra control.

Constraint forms (level L scales the baseline value of the policy p = 1, s = 0.75; metrics are daily means):
  hourly   driver hourly wage = driver income / online-vehicle minutes x 60  >= L x baseline
  budget   total driver income                                              >= L x baseline
  per_trip driver pay per unit of billable trip, sum(pay) / sum(k_d dist + k_t trip time) >= L, where
           k_d = 0.5 mean(pay)/mean(dist) and k_t = 0.5 mean(pay)/mean(time) from the baseline (ratio is 1 there)

Search (per policy): evolution strategy + coordinate pattern search, 3 independent restarts (the first from the
nested policy's optimum, so a larger family can never score below the smaller one it contains - asserted), scored
on 10 common-random-number days (a 3-day screen promotes ~40% of a population to the full 10 days).  Feasibility is
judged on those 10 days (relative shortfall penalty); no post-hoc projection.  The final comparison uses 20 fresh days
and reports, per policy, the share of days on which the constraint holds.
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
P_BOUNDS, S_BOUNDS = (0.8, 2.0), (0.1, 1.3)
BLOCKS, BLOCK_MINUTES = 4, 360
KINDS = {"f": 1, "a": 2, "a1": 5, "b": 9, "b1": 12}
DIMS = dict(KINDS, g=4, b4=5)  # (g) and (b4) are only used by the fixed-share / low-noise tools, not by the default study loop
FIXED_SHARE = 0.75  # kinds "f"/"g": prices optimised, share pinned (baseline 0.75; fixed_share_value tools override it)
LABELS = {"f": "(f) p, s = 0.75", "g": "(g) 4p, s fixed", "a": "(a) p + s", "a1": "(a') 4p + s", "b": "(b) p + 8s", "b4": "(b4) p + 4s", "b1": "(b') 4p + 8s"}
BUDGET = {"f": (6, 5), "g": (10, 6), "a": (6, 5), "a1": (10, 6), "b": (12, 7), "b4": (8, 6), "b1": (16, 8)}  # (population, generations) per restart
PENALTY = 2e5  # profit units per unit of relative constraint shortfall (1% shortfall = 2000)
BASELINE = (np.ones(BLOCKS), np.full(2 * BLOCKS, 0.75))
_CTX = {}


# ----------------------------------------------------------------------------- parameterisation
def canonical(kind, theta):
    """Free parameters -> (4 block prices, 8 shares ordered block-major, group-minor)."""
    theta = np.asarray(theta, dtype=float)
    if kind == "f":
        return np.full(BLOCKS, theta[0]), np.full(2 * BLOCKS, FIXED_SHARE)
    if kind == "g":
        return theta[:BLOCKS].copy(), np.full(2 * BLOCKS, FIXED_SHARE)
    if kind == "a":
        return np.full(BLOCKS, theta[0]), np.full(2 * BLOCKS, theta[1])
    if kind == "a1":
        return theta[:BLOCKS].copy(), np.full(2 * BLOCKS, theta[BLOCKS])
    if kind == "b":
        return np.full(BLOCKS, theta[0]), theta[1:].copy()
    if kind == "b4":  # one price all day, one share per block (both demand groups share it)
        return np.full(BLOCKS, theta[0]), np.repeat(theta[1:1 + BLOCKS], 2)
    return theta[:BLOCKS].copy(), theta[BLOCKS:].copy()


def embed(kind, prices, shares):
    """Canonical policy -> free parameters of ``kind`` (exact when the policy lies in the family)."""
    if kind == "f":
        return np.array([prices.mean()])
    if kind == "g":
        return np.array(prices, dtype=float)
    if kind == "a":
        return np.array([prices.mean(), shares.mean()])
    if kind == "a1":
        return np.r_[prices, shares.mean()]
    if kind == "b":
        return np.r_[prices.mean(), shares]
    if kind == "b4":
        return np.r_[prices.mean(), np.asarray(shares).reshape(BLOCKS, 2).mean(axis=1)]
    return np.r_[prices, shares]


def bounds(kind):
    d = DIMS[kind]
    n_p = {"f": 1, "g": BLOCKS, "a": 1, "a1": BLOCKS, "b": 1, "b4": 1, "b1": BLOCKS}[kind]
    lo = np.array([P_BOUNDS[0]] * n_p + [S_BOUNDS[0]] * (d - n_p))
    hi = np.array([P_BOUNDS[1]] * n_p + [S_BOUNDS[1]] * (d - n_p))
    scales = np.array([0.15] * n_p + [0.08] * (d - n_p))
    return lo, hi, scales, n_p


def make_policy(canon):
    prices, shares = np.asarray(canon[:BLOCKS]), np.asarray(canon[BLOCKS:]).reshape(BLOCKS, 2)
    group = _CTX["group"]
    base = copy.copy(_CTX["base"])
    base.constant = lambda t, prices=prices, shares=shares: (
        prices[min(int(t) // BLOCK_MINUTES, BLOCKS - 1)], shares[min(int(t) // BLOCK_MINUTES, BLOCKS - 1)][group])
    return base


def run_job(job):
    canon, seed = job
    scenario, market = _CTX["scenario"], _CTX["market"]
    policy = make_policy(canon)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=_CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    col = lib.METRICS.index
    served, minutes = trace[:, col("served")].sum(), trace[:, col("online_vehicles")].sum()
    idle = trace[:, col("driver_idle_cost")].sum()  # 0 unless --idle_cost_bearer driver
    income = trace[:, col("driver_total_income")].sum() - idle  # driver NET income (pay minus idle costs borne)
    return {"profit": trace[:, col("net_profit")].sum(), "revenue": trace[:, col("revenue")].sum(), "income": income, "idle_cost": idle,
            "wage": income / minutes * 60.0, "consumer_surplus": trace[:, col("consumer_surplus")].sum(), "served": served,
            "entered": trace[:, col("entered")].sum(), "wait": trace[:, col("wait_minutes")].sum() / served,
            "pickup": trace[:, col("pickup_minutes")].sum() / served, "empty_minutes": trace[:, col("empty_minutes")].sum(),
            "left": trace[:, col("left")].sum(), "online_minutes": minutes,
            "trip_distance": trace[:, col("trip_distance")].sum(), "trip_minutes": trace[:, col("trip_minutes")].sum()}


class Evaluator:
    """Parallel, memoised day evaluation of canonical policies."""

    def __init__(self, pool):
        self.pool, self.cache = pool, {}

    @staticmethod
    def key(canon):
        return tuple(np.round(np.asarray(canon, dtype=float), 9))

    def run(self, canons, seeds):
        missing = [(self.key(c), s) for c in canons for s in seeds if (self.key(c), s) not in self.cache]
        missing = list(dict.fromkeys(missing))
        if missing:
            for job, row in zip(missing, self.pool.map(run_job, missing, chunksize=1)):
                self.cache[job] = row
        return [[self.cache[(self.key(c), s)] for s in seeds] for c in canons]


class Constraint:
    def __init__(self, form, level, baseline_rows):
        self.form, self.level = form, level
        pay, served = sum(r["income"] for r in baseline_rows), sum(r["served"] for r in baseline_rows)
        dist, minutes = sum(r["trip_distance"] for r in baseline_rows), sum(r["trip_minutes"] for r in baseline_rows)
        self.k_d, self.k_t = 0.5 * pay / dist, 0.5 * pay / minutes
        base = {"hourly": np.mean([r["wage"] for r in baseline_rows]), "budget": np.mean([r["income"] for r in baseline_rows]),
                "per_trip": float(np.mean([self.raw(r) for r in baseline_rows]))}[form]
        self.baseline_value = base
        self.target = level * base

    def raw(self, row):
        return row["income"] / (self.k_d * row["trip_distance"] + self.k_t * row["trip_minutes"])

    def value(self, row):
        value = {"hourly": row["wage"], "budget": row["income"], "per_trip": self.raw(row)}[self.form]
        # a day with no online driver minutes (market collapse) has no wage: that is a violation, not a free pass
        return float(value) if np.isfinite(value) else 0.0

    def score(self, rows):
        profit = float(np.mean([r["profit"] for r in rows]))
        metric = float(np.mean([self.value(r) for r in rows]))
        shortfall = max(0.0, 1.0 - metric / self.target)
        return profit - PENALTY * shortfall, profit, metric


# ----------------------------------------------------------------------------- search
def full_objective(ev, con, canons, seeds):
    return np.array([con.score(rows)[0] for rows in ev.run(canons, seeds)])


def screened_objective(ev, con, canons, seeds, screen=3, promote=0.4):
    """3-day screen, then the full set of days for the best ~40% (the rest get -inf and are never selected)."""
    screen_scores = np.array([con.score(rows)[0] for rows in ev.run(canons, seeds[:screen])])
    keep = set(np.argsort(-screen_scores)[: max(2, int(np.ceil(promote * len(canons))))].tolist()) | {0}
    obj = np.full(len(canons), -np.inf)
    idx = sorted(keep)
    obj[idx] = full_objective(ev, con, [canons[i] for i in idx], seeds)
    return obj


def run_es(ev, con, kind, x0, lam, gens, sigma0, rng, seeds, log):
    lo, hi, scales, _ = bounds(kind)
    mean, sigma = np.clip(np.asarray(x0, dtype=float), lo, hi), sigma0
    best_x, best_f = mean.copy(), -np.inf
    mu = max(2, lam // 4)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    for g in range(gens):
        X = np.clip(mean + sigma * scales * rng.standard_normal((lam, len(mean))), lo, hi)
        X[0] = mean
        F = screened_objective(ev, con, [np.r_[canonical(kind, x)[0], canonical(kind, x)[1]] for x in X], seeds)
        order = np.argsort(-F)
        order = [i for i in order if np.isfinite(F[i])]
        improved = F[order[0]] > best_f + 1e-9
        if improved:
            best_f, best_x = float(F[order[0]]), X[order[0]].copy()
        top = order[:mu]
        w = weights[: len(top)] / weights[: len(top)].sum()
        mean = w @ X[top]
        sigma = float(np.clip(sigma * (1.15 if improved else 0.8), 0.05, 2.5))
    return best_x, best_f


def refine(ev, con, kind, x, f, seeds, log, rounds=2, max_iter=4):
    lo, hi, scales, _ = bounds(kind)
    step = scales * 0.5
    for _ in range(rounds):
        for _ in range(max_iter):
            neighbours = []
            for i in range(len(x)):
                for sign in (-1.0, 1.0):
                    cand = x.copy()
                    cand[i] = np.clip(cand[i] + sign * step[i], lo[i], hi[i])
                    neighbours.append(cand)
            F = full_objective(ev, con, [np.r_[canonical(kind, c)[0], canonical(kind, c)[1]] for c in neighbours], seeds)
            if F.max() > f + 1e-9:
                f, x = float(F.max()), neighbours[int(F.argmax())]
            else:
                break
        step = step / 2
    return x, f


def search(ev, con, kind, warm, rng, seeds, budget_scale, restarts, log):
    """Best of ``restarts`` ES runs plus the embedded warm-start policies (so nested optima are never lost)."""
    lam, gens = BUDGET[kind]
    gens = int(round(gens * budget_scale))
    lo, hi, scales, n_p = bounds(kind)
    candidates = []
    for prices, shares in warm:
        x = embed(kind, prices, shares)
        candidates.append((float(full_objective(ev, con, [np.r_[prices, shares]], seeds)[0]), x))
    runs = []
    for r in range(restarts):
        if r == 0 and candidates:
            x0, sigma0 = max(candidates, key=lambda c: c[0])[1], 1.0
        else:
            x0 = np.r_[rng.uniform(1.0, 1.6, n_p), rng.uniform(0.3, 1.0, DIMS[kind] - n_p)]
            sigma0 = 2.0
        x, f = run_es(ev, con, kind, x0, lam, gens, sigma0, rng, seeds, log)
        runs.append((f, x))
        log("  [{}] restart {} best objective {:.0f}".format(kind, r, f))
    best_f, best_x = max(candidates + runs, key=lambda c: c[0])
    best_x, best_f = refine(ev, con, kind, best_x, best_f, seeds, log)
    log("  [{}] final search objective {:.0f}".format(kind, best_f))
    return best_x, best_f, [f for f, _ in runs]


# ----------------------------------------------------------------------------- reporting
def summarize(rows, con):
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    values = np.array([con.value(r) for r in rows])
    out["constraint_value"] = float(values.mean())
    out["constraint_ratio"] = float(values.mean() / con.target)
    out["constraint_day_satisfaction"] = float(np.mean(values >= con.target))
    out["constraint_mean_satisfied"] = bool(values.mean() >= con.target)
    return out


def paired_delta(rows_b, rows_a, con=None, kappa=None):
    """Per-day (b - a): profit, allocation = -d(driver income), efficiency = d(profit) + d(income) and detail rows."""
    keys = list(rows_a[0])
    diff = {k: np.array([b[k] - a[k] for a, b in zip(rows_a, rows_b)]) for k in keys}
    diff["allocation"] = -diff["income"]
    diff["efficiency"] = diff["profit"] + diff["income"]
    # pay saved by fewer paid driver-minutes, at policy (a)'s mean pay per driver-minute
    pay_per_minute = np.mean([r["income"] for r in rows_a]) / np.mean([r["online_minutes"] for r in rows_a])
    diff["supply_channel"] = -pay_per_minute * diff["online_minutes"]
    if con is not None and kappa is not None:
        slack = np.array([(con.value(b) - con.value(a)) / con.target for a, b in zip(rows_a, rows_b)])
        diff["profit_constraint_adjusted"] = diff["profit"] + kappa * slack
    return {k: dict(zip(["mean", "lo", "hi"], lib.confidence_interval(v))) for k, v in diff.items()}


def run_setting(args, log):
    scenario = lib.Scenario(args.scenario)
    demand = scenario.dd_train.sum(axis=(0, 2))
    group = (demand > np.median(demand)).astype(int)
    overrides = dict(MARKET)
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value
    _CTX.update(scenario=scenario, market=lib.market_namespace({}, overrides), warmup=args.warmup, group=group,
                base=lib.constant_policy(scenario, 1.0, 0.75))
    opt_seeds = [args.opt_seed + r for r in range(args.opt_replicates)]
    final_seeds = [args.final_seed + r for r in range(args.final_replicates)]
    rng = np.random.default_rng(args.rng_seed)
    clock = time.time()
    stamp = lambda msg: log("{:>6.0f}s {}".format(time.time() - clock, msg))
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = Evaluator(pool)
        base_canon = np.r_[BASELINE[0], BASELINE[1]]
        baseline_rows = ev.run([base_canon], final_seeds)[0]
        con = Constraint(args.constraint, args.level, baseline_rows)
        stamp("== {} {} x{:g}: target {:.4g} (baseline {:.4g}) on {}".format(
            args.constraint, args.level, args.level, con.target, con.baseline_value, Path(args.scenario).name))
        sol, obj, restarts_obj = {}, {}, {}
        sol["f"], obj["f"], restarts_obj["f"] = search(ev, con, "f", [BASELINE], np.random.default_rng(args.rng_seed + 1),
                                                       opt_seeds, args.budget_scale, args.restarts, stamp)
        warm_f = canonical("f", sol["f"])
        sol["a"], obj["a"], restarts_obj["a"] = search(ev, con, "a", [BASELINE, warm_f], rng, opt_seeds, args.budget_scale, args.restarts, stamp)
        warm_a = canonical("a", sol["a"])
        sol["a1"], obj["a1"], restarts_obj["a1"] = search(ev, con, "a1", [warm_a], rng, opt_seeds, args.budget_scale, args.restarts, stamp)
        sol["b"], obj["b"], restarts_obj["b"] = search(ev, con, "b", [warm_a], rng, opt_seeds, args.budget_scale, args.restarts, stamp)
        sol["b1"], obj["b1"], restarts_obj["b1"] = search(ev, con, "b1", [canonical("a1", sol["a1"]), canonical("b", sol["b"])],
                                                          rng, opt_seeds, args.budget_scale, args.restarts, stamp)
        for big, small in [("a", "f"), ("a1", "a"), ("b", "a"), ("b1", "a1"), ("b1", "b")]:
            assert obj[big] >= obj[small] - 1e-6, "search objective of {} fell below {}".format(big, small)
        canon = {k: np.r_[canonical(k, sol[k])[0], canonical(k, sol[k])[1]] for k in KINDS}
        rows = {k: ev.run([canon[k]], final_seeds)[0] for k in KINDS}
        # marginal profit cost of one unit of relative constraint slack: +-0.02 uniform share shift of (a')
        shifted = [np.r_[canonical("a1", sol["a1"])[0], np.clip(canonical("a1", sol["a1"])[1] + off, *S_BOUNDS)] for off in (0.02, -0.02)]
        up, down = (ev.run([c], final_seeds)[0] for c in shifted)
        profit_slope = (np.mean([r["profit"] for r in up]) - np.mean([r["profit"] for r in down]))
        slack_slope = (np.mean([con.value(r) for r in up]) - np.mean([con.value(r) for r in down])) / con.target
        kappa = -profit_slope / slack_slope if abs(slack_slope) > 1e-9 else float("nan")
        lo_s, hi_s = S_BOUNDS
        result = {"scenario": args.scenario, "market": overrides, "constraint": args.constraint, "level": args.level,
                  "target": con.target, "baseline_value": con.baseline_value, "k_d": con.k_d, "k_t": con.k_t,
                  "group": group.tolist(), "opt_seeds": opt_seeds, "final_seeds": final_seeds, "budget_scale": args.budget_scale,
                  "restarts": args.restarts, "search_objective": obj, "restart_objectives": restarts_obj, "kappa_profit_per_relative_slack": kappa,
                  "baseline": summarize(baseline_rows, con), "policies": {}, "deltas": {}}
        for k in KINDS:
            prices, shares = canonical(k, sol[k])
            summary = summarize(rows[k], con)
            summary.update(prices=prices.tolist(), shares=shares.tolist(),
                           shares_at_bound=bool(np.any(shares <= lo_s + 0.005) or np.any(shares >= hi_s - 0.005)),
                           prices_at_bound=bool(np.any(prices <= P_BOUNDS[0] + 0.005) or np.any(prices >= P_BOUNDS[1] - 0.005)))
            result["policies"][k] = summary
        for name, (big, small) in {"b1_vs_a1": ("b1", "a1"), "a_vs_f": ("a", "f"), "b_vs_a": ("b", "a"), "a1_vs_a": ("a1", "a"), "b1_vs_b": ("b1", "b")}.items():
            result["deltas"][name] = paired_delta(rows[big], rows[small], con, kappa)
        d = result["deltas"]["b1_vs_a1"]["profit"]
        stamp("  Delta (b')-(a') = {:+.0f} [{:+.0f}, {:+.0f}]; (b)-(a) = {:+.0f}".format(d["mean"], d["lo"], d["hi"], result["deltas"]["b_vs_a"]["profit"]["mean"]))
        result["wall_seconds"] = time.time() - clock
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--constraint", choices=["hourly", "budget", "per_trip"], default="hourly")
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--market", default="", help="overrides k=v,... e.g. beta_driver=0.3,online_zone_choice=pay_weighted")
    parser.add_argument("--opt_replicates", type=int, default=10)
    parser.add_argument("--final_replicates", type=int, default=20)
    parser.add_argument("--opt_seed", type=int, default=4000)
    parser.add_argument("--final_seed", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=480)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--budget_scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--rng_seed", type=int, default=12345)
    args = parser.parse_args(argv)
    result = run_setting(args, lambda m: print(m, flush=True))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
