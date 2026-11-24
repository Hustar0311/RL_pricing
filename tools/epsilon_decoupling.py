"""Epsilon-constraint comparison of the policy classes (a) p + s and (b) p + 8 shares (uniform price).

For a driver-metric level W (relative to the p = 1, s = 0.75 reference) each class solves
    max  Profit - mu * max(0, W - G)^2 - mu * max(0, G - W - band)^2          (G = metric / reference, band = 0.02)
so the solution has G in [W, W + band].  The Lagrangian frontier (frontier_decoupling.py) only reaches the concave hull;
this reaches non-concave stretches.  Modes
  default   one (a) and one (b) per target, warm-started from the nearest Lagrangian-frontier points (--warm_from)
  --causal  also solves (b) with an online-minutes floor equal to the (a) solution's online minutes at the same target
  --repeats N   N independent searches per target and class, each started from the reference policy only ((b) from its
            own (a)); reports the distribution of Delta = Profit(b) - Profit(a)
(b) contains (a) and the evaluation days are new, so Delta is a lower bound on the true gain; negative values mean
"no gain detected".
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import static_decoupling as dec  # noqa: E402
from tools import frontier_decoupling as fr  # noqa: E402

BAND = 0.02
MU = 1e7


class Epsilon(dec.Constraint):
    """Quadratic penalty keeping G / G_base inside [target, target + band]; optional online-minutes floor."""

    def __init__(self, form, baseline_rows, target, mu=MU, band=BAND, om_floor=None, mu_om=MU):
        super().__init__(form, 1.0, baseline_rows)
        self.t, self.mu, self.band, self.om_floor, self.mu_om = float(target), mu, band, om_floor, mu_om

    def score(self, rows):
        profit = float(np.mean([r["profit"] for r in rows]))
        metric = float(np.mean([self.value(r) for r in rows]))
        g = metric / self.baseline_value
        pen = self.mu * (max(0.0, self.t - g) ** 2 + max(0.0, g - self.t - self.band) ** 2)
        if self.om_floor is not None:
            om = float(np.mean([r["online_minutes"] for r in rows]))
            pen += self.mu_om * max(0.0, 1.0 - om / self.om_floor) ** 2
        return profit - pen, profit, metric


def boot_mean(x, n_boot=2000, seed=0):
    """Mean and 95% interval of per-day values over resampled days."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return {"mean": float(x.mean()), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def nearest(points, kind, g_target, k=2):
    """Stored (prices, shares) of the k Lagrangian-frontier points whose G is nearest to the target."""
    order = sorted(points, key=lambda e: abs(np.mean(e[kind]["days"]["g_rel"]) - g_target))
    return [(np.array(e[kind]["prices"]), np.array(e[kind]["shares"])) for e in order[:k]]


def evaluate(ev, con, kind, x, final_seeds):
    rows = fr.point_rows(ev, con, kind, x, final_seeds)
    prices, shares = dec.canonical(kind, x)
    return {"prices": prices.tolist(), "shares": shares.tolist(), "days": fr.day_arrays(rows, con), "summary": dec.summarize(rows, con)}


def mean_of(entry, key):
    return float(np.mean(entry["days"][key]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--form", choices=["hourly", "budget"], required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--targets", required=True, help="comma separated W values (relative to the reference)")
    ap.add_argument("--causal_targets", default="", help="subset of targets that also get the online-minutes-floor (b)")
    ap.add_argument("--warm_from", default=None, help="frontier_decoupling JSON whose points warm-start the searches")
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--opt_replicates", type=int, default=10)
    ap.add_argument("--final_replicates", type=int, default=20)
    ap.add_argument("--opt_seed", type=int, default=4000)
    ap.add_argument("--final_seed", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=480)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--budget_scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--rng_seed", type=int, default=24680)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    overrides, group = fr.setup(args)
    targets = [float(v) for v in args.targets.split(",")]
    causal = {float(v) for v in args.causal_targets.split(",") if v}
    opt_seeds = [args.opt_seed + r for r in range(args.opt_replicates)]
    final_seeds = [args.final_seed + r for r in range(args.final_replicates)]
    clock = time.time()
    log = lambda msg: print("{:>6.0f}s {}".format(time.time() - clock, msg), flush=True)
    frontier = json.loads(Path(args.warm_from).read_text(encoding="utf-8"))["points"] if args.warm_from else []
    result = {"scenario": args.scenario, "market": overrides, "form": args.form, "mu": args.mu, "band": BAND, "restarts": args.restarts,
              "opt_seeds": opt_seeds, "final_seeds": final_seeds, "points": [], "repeats": []}
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = dec.Evaluator(pool)
        baseline_rows = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], final_seeds)[0]
        mk = lambda t, **kw: Epsilon(args.form, baseline_rows, t, mu=args.mu, **kw)
        result["baseline_value"] = mk(1.0).baseline_value
        result["baseline"] = dec.summarize(baseline_rows, mk(1.0))
        log("== {} epsilon on {}: G_base {:.4g}".format(args.form, Path(args.scenario).name, result["baseline_value"]))
        rng = np.random.default_rng(args.rng_seed)
        prev = {"a": None, "b": None}
        for t in ([] if args.repeats else targets):  # --repeats replaces the warm-started pass
            con = mk(t)
            warm_a = [dec.BASELINE] + (nearest(frontier, "a", t) if frontier else []) + ([prev["a"]] if prev["a"] else [])
            xa, fa, _ = dec.search(ev, con, "a", warm_a, rng, opt_seeds, args.budget_scale, args.restarts, log)
            ca = dec.canonical("a", xa)
            warm_b = [ca] + (nearest(frontier, "b", t) if frontier else []) + ([prev["b"]] if prev["b"] else [])
            xb, fb, _ = dec.search(ev, con, "b", warm_b, rng, opt_seeds, args.budget_scale, args.restarts, log)
            prev = {"a": ca, "b": dec.canonical("b", xb)}
            entry = {"target": t, "objective_a": fa, "objective_b": fb, "a": evaluate(ev, con, "a", xa, final_seeds),
                     "b": evaluate(ev, con, "b", xb, final_seeds)}
            if t in causal:
                om_ref = float(np.mean([r["online_minutes"] for r in fr.point_rows(ev, con, "a", xa, opt_seeds)]))
                con_c = mk(t, om_floor=om_ref)
                xc, fc, _ = dec.search(ev, con_c, "b", [ca, dec.canonical("b", xb)], rng, opt_seeds, args.budget_scale, args.restarts, log)
                entry["b_floor"] = evaluate(ev, con, "b", xc, final_seeds)
                entry["om_floor"] = om_ref
                entry["objective_b_floor"] = fc
            result["points"].append(entry)
            d = np.array(entry["b"]["days"]["profit"]) - np.array(entry["a"]["days"]["profit"])
            log("  target {:.2f}: (a) G {:.3f} profit {:.0f} | (b) G {:.3f} profit {:.0f} | delta {:+.0f}{}".format(
                t, mean_of(entry["a"], "g_rel"), mean_of(entry["a"], "profit"), mean_of(entry["b"], "g_rel"), mean_of(entry["b"], "profit"),
                d.mean(), "" if "b_floor" not in entry else " | floor-(b) profit {:.0f} G {:.3f}".format(
                    mean_of(entry["b_floor"], "profit"), mean_of(entry["b_floor"], "g_rel"))))
        for t in targets if args.repeats else []:
            con, reps = mk(t), []
            for rep in range(args.repeats):
                rep_rng = np.random.default_rng(args.rng_seed + 100 + rep)
                xa, fa, _ = dec.search(ev, con, "a", [dec.BASELINE], rep_rng, opt_seeds, args.budget_scale, args.restarts, log)
                xb, fb, _ = dec.search(ev, con, "b", [dec.canonical("a", xa)], rep_rng, opt_seeds, args.budget_scale, args.restarts, log)
                ea, eb = evaluate(ev, con, "a", xa, final_seeds), evaluate(ev, con, "b", xb, final_seeds)
                reps.append({"objective_a": fa, "objective_b": fb, "a": ea, "b": eb})
                log("  repeat {} target {:.2f}: delta {:+.0f} (G a {:.3f} b {:.3f})".format(
                    rep, t, mean_of(eb, "profit") - mean_of(ea, "profit"), mean_of(ea, "g_rel"), mean_of(eb, "g_rel")))
            result["repeats"].append({"target": t, "reps": reps})
    result["wall_seconds"] = time.time() - clock
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("done")
    return result


if __name__ == "__main__":
    main()
