"""Profit-vs-driver-metric frontier of the uniform-price policy classes (a) p + s and (b) p + 8 shares (no price blocks).

For a driver metric G (hourly wage or total driver income, both net of idle costs borne by drivers) and a multiplier
lambda, each frontier point solves  max  Profit + lambda * G / G_base  (G_base = value at the reference policy p = 1,
s = 0.75; lambda is in profit units per unit of relative G).  This replaces the post-hoc "constraint adjustment":
Delta(m) = Profit_b - Profit_a is read off the two frontiers at G = m * G_base by linear interpolation of the 20-day
evaluation means, with a bootstrap over evaluation days (the same days for every point, so all curves are paired).

  1. pilot: (a) with a small budget at a coarse lambda list -> lambda grid whose (a) points hit G = 0.8 ... 1.25 G_base
  2. optimiser variance (optional): independent searches of (a) and (b) at chosen grid points; best repeats are fed
     back as warm-start candidates
  3. frontier: ascending lambda; (a) warm-started from the previous lambda; (b) from (a) at the same lambda (its
     solution is always a candidate) and from the previous (b); best of restarts + warm-start candidates
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import static_decoupling as dec  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402

PILOT_LAMBDAS = [0.0, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 1.2e5, 1.5e5]
TARGETS = [0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.25]
MULTS = [0.8, 1.0, 1.2]


class Lagrangian(dec.Constraint):
    """Objective Profit + lambda * G / G_base on the optimisation days (the base class supplies G and G_base)."""

    def __init__(self, form, baseline_rows, lam):
        super().__init__(form, 1.0, baseline_rows)
        self.lam = float(lam)

    def score(self, rows):
        profit = float(np.mean([r["profit"] for r in rows]))
        metric = float(np.mean([self.value(r) for r in rows]))
        return profit + self.lam * metric / self.baseline_value, profit, metric


# ----------------------------------------------------------------------------- frontier arithmetic
def interpolate(g_rel, profit, grid):
    """Profit of one frontier at G = grid (linear in G between the points, NaN outside their range)."""
    order = np.argsort(g_rel)
    x = np.asarray(g_rel, dtype=float)[order]
    grid = np.asarray(grid, dtype=float)
    snapped = np.where(np.abs(grid - x[0]) < 1e-9, x[0], np.where(np.abs(grid - x[-1]) < 1e-9, x[-1], grid))  # float edge
    return np.interp(snapped, x, np.asarray(profit)[order], left=np.nan, right=np.nan)


def delta_curve(g_a, p_a, g_b, p_b, grid):
    """g_*/p_*: arrays (points, days).  Profit_b - Profit_a at each grid value, using the day-mean of every point."""
    return interpolate(g_b.mean(axis=1), p_b.mean(axis=1), grid) - interpolate(g_a.mean(axis=1), p_a.mean(axis=1), grid)


def bootstrap_delta(g_a, p_a, g_b, p_b, grid, n_boot=2000, seed=0):
    """Point estimate and 95% bootstrap interval of Delta over resampled evaluation days (paired across points)."""
    rng = np.random.default_rng(seed)
    days = g_a.shape[1]
    point = delta_curve(g_a, p_a, g_b, p_b, grid)
    draws = np.empty((n_boot, len(grid)))
    for i in range(n_boot):
        idx = rng.integers(0, days, days)
        draws[i] = delta_curve(g_a[:, idx], p_a[:, idx], g_b[:, idx], p_b[:, idx], grid)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = (np.nanpercentile(draws, q, axis=0) for q in (2.5, 97.5))
    return point, lo, hi


def choose_lambdas(pilot, targets=TARGETS):
    """Lambda values whose pilot (a) points reach the targets: linear interpolation of lambda over the pilot G."""
    lam, g = np.array([p[0] for p in pilot]), np.array([p[1] for p in pilot])
    order = np.argsort(lam)
    lam, g = lam[order], np.maximum.accumulate(g[order])  # G(lambda) is non-decreasing; remove search noise
    keep = np.r_[True, np.diff(g) > 1e-9]  # drop saturated plateaus
    lam, g = lam[keep], g[keep]
    chosen = np.interp(targets, g, lam)  # targets outside the reached range clip to the end points
    return sorted(set(round(float(x), 1) for x in chosen))



def fill_lambdas(result, max_new=4, jump=0.2):
    """Extra lambdas for a finished run: beyond the grid ends when a frontier misses a target multiplier, and at the
    midpoint of adjacent grid lambdas whose G jumps by more than `jump` (a non-concave stretch the Lagrangian skips)."""
    pts = sorted(result["points"], key=lambda p: p["lambda"])
    lam = np.array([p["lambda"] for p in pts])
    ga = np.array([np.mean(p["a"]["days"]["g_rel"]) for p in pts])
    gb = np.array([np.mean(p["b"]["days"]["g_rel"]) for p in pts])
    new = []
    # a frontier that stops short of a target (the pair is paired on G, so the weaker side decides) -> push lambda beyond the grid
    if min(ga.max(), gb.max()) < max(MULTS):
        new += [lam.max() * f for f in (1.05, 1.12, 1.25, 1.5)]
    if max(ga.min(), gb.min()) > min(MULTS):
        new += [lam.min() * f for f in (0.9, 0.75, 0.5)]
    for g in (ga, gb):
        for i in np.where(np.diff(g) > jump)[0]:
            new.append(float((lam[i] + lam[i + 1]) / 2))
    new = sorted(set(round(x, 1) for x in new if min(abs(x - lam)) > 1.0))
    if len(new) > max_new:
        new = sorted(new[i] for i in np.linspace(0, len(new) - 1, max_new).round().astype(int))
    return new


def finish(result, points, args, log):
    """Delta(m) with bootstrap bands and the per-point service metrics from the evaluation days of every point."""
    arr = lambda kind, key: np.array([p[kind]["days"][key] for p in points])
    g_a, p_a, g_b, p_b = arr("a", "g_rel"), arr("a", "profit"), arr("b", "g_rel"), arr("b", "profit")
    point, lo, hi = bootstrap_delta(g_a, p_a, g_b, p_b, MULTS, args.bootstrap)
    result["delta_at"] = {str(m): {"mean": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, m in enumerate(MULTS)}
    grid = np.linspace(0.8, 1.25, 19)
    cp, clo, chi = bootstrap_delta(g_a, p_a, g_b, p_b, grid, args.bootstrap)
    result["delta_curve"] = {"grid": grid.tolist(), "mean": cp.tolist(), "lo": clo.tolist(), "hi": chi.tolist()}
    for key in ("online_minutes", "served", "consumer_surplus"):
        result["delta_curve_" + key] = {}
        ma, mb = arr("a", key), arr("b", key)
        for tag, m, g in (("a", ma, g_a), ("b", mb, g_b)):
            result["delta_curve_" + key][tag] = {"g_rel": g.mean(axis=1).tolist(), "mean": m.mean(axis=1).tolist()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("Delta at {}: {}".format(MULTS, [tuple(np.round([result["delta_at"][str(m)][k] for k in ("mean", "lo", "hi")], 0).tolist()) for m in MULTS]))


# ----------------------------------------------------------------------------- driver
def setup(args):
    scenario = lib.Scenario(args.scenario)
    demand = scenario.dd_train.sum(axis=(0, 2))
    group = (demand > np.median(demand)).astype(int)
    overrides = dict(dec.MARKET)
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value
    dec._CTX.update(scenario=scenario, market=lib.market_namespace({}, overrides), warmup=args.warmup, group=group,
                    base=lib.constant_policy(scenario, 1.0, 0.75))
    return overrides, group


def point_rows(ev, con, kind, x, seeds):
    canon = np.r_[dec.canonical(kind, x)[0], dec.canonical(kind, x)[1]]
    return ev.run([canon], seeds)[0]


def day_arrays(rows, con):
    keys = {"profit": "profit", "online_minutes": "online_minutes", "served": "served", "consumer_surplus": "consumer_surplus",
            "income": "income", "wage": "wage"}
    out = {k: [float(r[v]) for r in rows] for k, v in keys.items()}
    out["g_rel"] = [con.value(r) / con.baseline_value for r in rows]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--form", choices=["hourly", "budget"], required=True)
    parser.add_argument("--market", default="")
    parser.add_argument("--lambdas", default="auto", help="'auto' (pilot) or comma separated lambda values")
    parser.add_argument("--variance_reps", type=int, default=0)
    parser.add_argument("--variance_idx", default="3,4", help="indices into the lambda grid used for the variance study")
    parser.add_argument("--opt_replicates", type=int, default=10)
    parser.add_argument("--final_replicates", type=int, default=20)
    parser.add_argument("--opt_seed", type=int, default=4000)
    parser.add_argument("--final_seed", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=480)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--budget_scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--rng_seed", type=int, default=12345)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reuse", default=None, help="finished frontier JSON: keep its points, solve only the extra lambdas")
    parser.add_argument("--extra_lambdas", default="", help="comma separated lambdas (or 'auto') added to the --reuse run")
    args = parser.parse_args(argv)
    overrides, group = setup(args)
    opt_seeds = [args.opt_seed + r for r in range(args.opt_replicates)]
    final_seeds = [args.final_seed + r for r in range(args.final_replicates)]
    clock = time.time()
    log = lambda msg: print("{:>6.0f}s {}".format(time.time() - clock, msg), flush=True)
    result = {"scenario": args.scenario, "market": overrides, "form": args.form, "restarts": args.restarts,
              "opt_seeds": opt_seeds, "final_seeds": final_seeds, "group": group.tolist()}
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = dec.Evaluator(pool)
        baseline_rows = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], final_seeds)[0]
        lag = lambda lam: Lagrangian(args.form, baseline_rows, lam)
        g_base = lag(0.0).baseline_value
        result["baseline_value"] = g_base
        result["baseline"] = dec.summarize(baseline_rows, lag(0.0))
        log("== {} frontier on {}: G_base {:.4g}".format(args.form, Path(args.scenario).name, g_base))
        rng = np.random.default_rng(args.rng_seed)

        if args.reuse:  # add lambdas to a finished run: its points stay, new ones warm-start from the nearest stored point
            old = json.loads(Path(args.reuse).read_text(encoding="utf-8"))
            assert old["form"] == args.form and old["final_seeds"] == final_seeds and old["opt_seeds"] == opt_seeds, "reuse: settings differ"
            assert abs(old["baseline_value"] - g_base) < 1e-9 * max(1.0, abs(g_base)), "reuse: baseline differs"
            extra_lams = fill_lambdas(old) if args.extra_lambdas == "auto" else [float(v) for v in args.extra_lambdas.split(",") if v]
            log("reusing {} points; new lambdas {}".format(len(old["points"]), extra_lams))
            points = list(old["points"])
            stored = lambda entry, kind: (np.array(entry[kind]["prices"]), np.array(entry[kind]["shares"]))
            for lam in extra_lams:
                con = lag(lam)
                order = sorted(points, key=lambda e: abs(e["lambda"] - lam))
                warm_a = [stored(e, "a") for e in order[:2]]
                xa, fa, _ = dec.search(ev, con, "a", warm_a, rng, opt_seeds, args.budget_scale, args.restarts, log)
                warm_b = [dec.canonical("a", xa)] + [stored(e, "b") for e in order[:2]]
                xb, fb, _ = dec.search(ev, con, "b", warm_b, rng, opt_seeds, args.budget_scale, args.restarts, log)
                assert fb >= fa - 1e-6, "(b) fell below (a) at lambda {}".format(lam)
                entry = {"lambda": lam, "objective_a": fa, "objective_b": fb, "added": True}
                for kind, x in (("a", xa), ("b", xb)):
                    rows = point_rows(ev, con, kind, x, final_seeds)
                    prices, shares = dec.canonical(kind, x)
                    entry[kind] = {"prices": prices.tolist(), "shares": shares.tolist(), "days": day_arrays(rows, con),
                                   "summary": dec.summarize(rows, con)}
                points.append(entry)
                log("  lambda {:.0f}: (a) G {:.3f} profit {:.0f} | (b) G {:.3f} profit {:.0f}".format(
                    lam, np.mean(entry["a"]["days"]["g_rel"]), np.mean(entry["a"]["days"]["profit"]),
                    np.mean(entry["b"]["days"]["g_rel"]), np.mean(entry["b"]["days"]["profit"])))
            points.sort(key=lambda e: e["lambda"])
            old["points"], old["lambdas"] = points, [e["lambda"] for e in points]
            old["reused_from"] = str(args.reuse)
            old["wall_seconds_extension"] = time.time() - clock
            finish(old, points, args, log)
            return old

        # 1. lambda grid
        if args.lambdas == "auto":
            pilot, warm = [], [dec.BASELINE]
            for lam in PILOT_LAMBDAS:
                con = lag(lam)
                x, f, _ = dec.search(ev, con, "a", warm, np.random.default_rng(args.rng_seed + 7), opt_seeds, 1.0, 2, lambda m: None)
                warm = [dec.canonical("a", x)]
                g = con.score(point_rows(ev, con, "a", x, opt_seeds))[2] / g_base
                pilot.append((lam, g))
                log("  pilot lambda {:.0f}: G/G_base {:.3f}".format(lam, g))
            lambdas = choose_lambdas(pilot)
            result["pilot"] = pilot
        else:
            lambdas = [float(v) for v in args.lambdas.split(",")]
        result["lambdas"] = lambdas
        log("lambda grid {}".format(lambdas))

        # 2. optimiser variance at selected grid points
        extra = {}
        result["variance"] = {}
        if args.variance_reps > 0:
            for idx in [int(i) for i in args.variance_idx.split(",")]:
                lam, con = lambdas[idx], lag(lambdas[idx])
                reps = []
                for rep in range(args.variance_reps):
                    rep_rng = np.random.default_rng(args.rng_seed + 100 + rep)
                    xa, fa, _ = dec.search(ev, con, "a", [dec.BASELINE], rep_rng, opt_seeds, args.budget_scale, args.restarts, lambda m: None)
                    xb, fb, _ = dec.search(ev, con, "b", [dec.canonical("a", xa)], rep_rng, opt_seeds, args.budget_scale, args.restarts, lambda m: None)
                    fin = {}
                    for kind, x in (("a", xa), ("b", xb)):
                        rows = point_rows(ev, con, kind, x, final_seeds)
                        fin[kind] = {"profit": float(np.mean([r["profit"] for r in rows])),
                                     "g_rel": float(np.mean([con.value(r) for r in rows]) / g_base)}
                    reps.append({"objective_a": fa, "objective_b": fb, "final": fin, "x_a": xa.tolist(), "x_b": xb.tolist()})
                    log("  variance idx {} lambda {:.0f} rep {}: obj a {:.0f} b {:.0f}".format(idx, lam, rep, fa, fb))
                result["variance"][str(idx)] = {"lambda": lam, "reps": reps}
                best_a, best_b = max(reps, key=lambda r: r["objective_a"]), max(reps, key=lambda r: r["objective_b"])
                extra[idx] = {"a": [dec.canonical("a", np.array(best_a["x_a"]))], "b": [dec.canonical("b", np.array(best_b["x_b"]))]}

        # 3. frontier
        points = []
        prev = {"a": None, "b": None}
        for i, lam in enumerate(lambdas):
            con = lag(lam)
            warm_a = ([dec.canonical("a", prev["a"])] if prev["a"] is not None else [dec.BASELINE]) + extra.get(i, {}).get("a", [])
            xa, fa, _ = dec.search(ev, con, "a", warm_a, rng, opt_seeds, args.budget_scale, args.restarts, log)
            warm_b = [dec.canonical("a", xa)] + ([dec.canonical("b", prev["b"])] if prev["b"] is not None else []) + extra.get(i, {}).get("b", [])
            xb, fb, _ = dec.search(ev, con, "b", warm_b, rng, opt_seeds, args.budget_scale, args.restarts, log)
            assert fb >= fa - 1e-6, "(b) fell below (a) at lambda {}".format(lam)
            prev = {"a": xa, "b": xb}
            entry = {"lambda": lam, "objective_a": fa, "objective_b": fb}
            for kind, x in (("a", xa), ("b", xb)):
                rows = point_rows(ev, con, kind, x, final_seeds)
                prices, shares = dec.canonical(kind, x)
                entry[kind] = {"prices": prices.tolist(), "shares": shares.tolist(), "days": day_arrays(rows, con),
                               "summary": dec.summarize(rows, con)}
            points.append(entry)
            log("  lambda {:.0f}: (a) G {:.3f} profit {:.0f} | (b) G {:.3f} profit {:.0f}".format(
                lam, np.mean(entry["a"]["days"]["g_rel"]), np.mean(entry["a"]["days"]["profit"]),
                np.mean(entry["b"]["days"]["g_rel"]), np.mean(entry["b"]["days"]["profit"])))
        result["points"] = points

    result["wall_seconds"] = time.time() - clock
    finish(result, points, args, log)
    return result


if __name__ == "__main__":
    main()
