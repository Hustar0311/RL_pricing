"""Exact-constraint solver: every candidate policy is moved onto the constraint G = W (no penalty function).

For a driver metric G (hourly wage or total driver income) and a target W = W_bar * G_base the constraint is met with
equality by solving for the driver share level s_bar, so the optimiser only searches the remaining parameters:
  (a)   candidate = price p; s solves G(p, s) = W   (p on a grid: 0.1 steps over [0.8, 2.0], then 0.025 steps within
        +-0.1 of the best coarse price -> the 0.025 grid is exact around the optimum)
  (b4)  candidate = (p, d1, d2, d3); the block shares are s_k = s_bar + d_k with sum_k w_k d_k = 0 (w_k = share of the
        baseline's online minutes in block k, d4 follows); s_bar solves G = W.  Evolution strategy over (p, d1, d2, d3);
        the (a) solution (d = 0) is always a candidate, so (b4) >= (a) on the search days.
s_bar is found by a safeguarded secant / regula-falsi iteration, evaluated in lock-step for all candidates on the same
common-random-number days (|G/W - 1| < 0.3%), and the profit at G = W is read off the linear interpolation.
Monotonicity of G in s is verified first on a grid of (p, d) points; if it fails, the solver switches to a fine s scan
(step 0.025, best crossing).  Each repeat uses its own 30 search days; solutions are evaluated on 20 new days.
The baseline (p = 1, s = 0.75) defining G_base is evaluated on the same days as the search (in-sample, constraint met
exactly there); on the evaluation days G is reported relative to the evaluation-day baseline.
usage: exact_constraint.py --scenario S --market k=v --form hourly|budget --targets 1.0,1.2 --repeats 5 --output out.json
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

S_LO, S_HI = dec.S_BOUNDS
TOL = 0.003
SLOPE0 = 1.4  # typical d(G/W)/ds, only for the first step
MAX_ITER = 7


def canon_of(p, d, s_bar):
    return np.r_[np.full(dec.BLOCKS, p), np.repeat(s_bar + np.asarray(d, dtype=float), 2)]


def s_range(d):
    d = np.asarray(d, dtype=float)
    return S_LO - d.min(), S_HI - d.max()


class Solver:
    def __init__(self, ev, con, seeds, log):
        self.ev, self.con, self.seeds, self.log = ev, con, seeds, log
        self.monotone = True
        self.evals = 0

    def measure(self, jobs):
        """jobs: list of (p, d, s_bar) -> arrays of (G / W, profit) on the search days."""
        rows = self.ev.run([canon_of(*j) for j in jobs], self.seeds)
        self.evals += len(jobs)
        ratio = np.array([np.mean([self.con.value(r) for r in rs]) / self.con.target for rs in rows])
        profit = np.array([np.mean([r["profit"] for r in rs]) for rs in rows])
        return ratio, profit

    def solve(self, cands):
        """cands: list of dict(p, d, s0).  Returns a list of dict(ok, s, profit, ratio, iters) (ok=False: unreachable)."""
        if not self.monotone:
            return [self.scan(c) for c in cands]
        pts = [[] for _ in cands]
        state = ["run"] * len(cands)
        rng_s = [s_range(c["d"]) for c in cands]
        nxt = [float(np.clip(c["s0"], *rng_s[i])) for i, c in enumerate(cands)]
        for it in range(MAX_ITER):
            active = [i for i, st in enumerate(state) if st == "run"]
            if not active:
                break
            ratio, profit = self.measure([(cands[i]["p"], cands[i]["d"], nxt[i]) for i in active])
            for k, i in enumerate(active):
                pts[i].append((nxt[i], float(ratio[k]), float(profit[k])))
                lo, hi = rng_s[i]
                s, r, _ = pts[i][-1]
                if abs(r - 1.0) < TOL:
                    state[i] = "ok"
                    continue
                if (s >= hi - 1e-9 and r < 1.0) or (s <= lo + 1e-9 and r > 1.0):
                    state[i] = "bound"  # the constraint level is unreachable inside the share bounds
                    continue
                nxt[i] = self.step(pts[i], lo, hi)
                if any(abs(nxt[i] - q[0]) < 1e-6 for q in pts[i]):
                    state[i] = "stall"
        out = []
        for i, c in enumerate(cands):
            out.append(self.finish(pts[i], state[i], rng_s[i], c["s0"]))
        return out

    @staticmethod
    def step(points, lo, hi):
        below = [q for q in points if q[1] < 1.0]
        above = [q for q in points if q[1] > 1.0]
        if below and above:  # regula falsi on the tightest bracket
            a, b = max(below, key=lambda q: q[1]), min(above, key=lambda q: q[1])
            s = a[0] + (1.0 - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        else:
            q = min(points, key=lambda q: abs(q[1] - 1.0))
            slope = SLOPE0
            others = [o for o in points if o is not q and abs(o[0] - q[0]) > 1e-9]
            if others:
                o = min(others, key=lambda o: abs(o[1] - 1.0))
                est = (q[1] - o[1]) / (q[0] - o[0])
                if np.isfinite(est) and est > 0.2:
                    slope = est
            s = q[0] + float(np.clip((1.0 - q[1]) / slope, -0.3, 0.3))
        return float(np.clip(s, lo, hi))

    @staticmethod
    def finish(points, state, srange, s0):
        if not points:
            return {"ok": False, "s": s0, "profit": -np.inf, "ratio": np.nan, "iters": 0}
        q = min(points, key=lambda q: abs(q[1] - 1.0))
        ok = abs(q[1] - 1.0) < TOL * 1.5
        s, profit = q[0], q[2]
        others = [o for o in points if o is not q and abs(o[1] - q[1]) > 1e-9]
        if ok and others:  # linear interpolation / tiny extrapolation to G = W
            o = min(others, key=lambda o: abs(o[1] - 1.0))
            f = (1.0 - q[1]) / (o[1] - q[1])
            s, profit = q[0] + f * (o[0] - q[0]), q[2] + f * (o[2] - q[2])
        return {"ok": bool(ok), "s": float(np.clip(s, *srange)), "profit": float(profit) if ok else -np.inf,
                "ratio": q[1], "iters": len(points), "state": state}

    def scan(self, c, step=0.025):
        lo, hi = s_range(c["d"])
        grid = np.arange(lo, hi + 1e-9, step)
        ratio, profit = self.measure([(c["p"], c["d"], s) for s in grid])
        best = {"ok": False, "s": c["s0"], "profit": -np.inf, "ratio": np.nan, "iters": len(grid)}
        for k in range(len(grid) - 1):
            if (ratio[k] - 1.0) * (ratio[k + 1] - 1.0) <= 0 and ratio[k] != ratio[k + 1]:
                f = (1.0 - ratio[k]) / (ratio[k + 1] - ratio[k])
                value = profit[k] + f * (profit[k + 1] - profit[k])
                if value > best["profit"]:
                    best = {"ok": True, "s": float(grid[k] + f * step), "profit": float(value), "ratio": 1.0, "iters": len(grid)}
        return best

    def check_monotone(self, points):
        """G(s) must be non-decreasing on a 0.1 grid of s for every (p, d) in points; returns the report."""
        report = []
        for p, d in points:
            lo, hi = s_range(d)
            grid = np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1)
            ratio, _ = self.measure([(p, d, s) for s in grid])
            inc = bool(np.all(np.diff(ratio) > -0.002))
            report.append({"p": p, "d": list(map(float, d)), "s": grid.round(3).tolist(), "ratio": ratio.round(4).tolist(), "monotone": inc})
        self.monotone = all(r["monotone"] for r in report)
        return report


def search_a(solver):
    coarse = [round(0.8 + 0.1 * i, 3) for i in range(13)]
    solved = {}
    res = solver.solve([{"p": p, "d": np.zeros(4), "s0": 0.75} for p in coarse])
    solved.update({p: r for p, r in zip(coarse, res)})
    best = max(solved, key=lambda p: solved[p]["profit"])
    fine = [round(best + 0.025 * k, 3) for k in range(-3, 4) if k % 4 and 0.8 - 1e-9 <= best + 0.025 * k <= 2.0 + 1e-9]
    fine = [p for p in fine if p not in solved]
    res = solver.solve([{"p": p, "d": np.zeros(4), "s0": solved[best]["s"]} for p in fine])
    solved.update({p: r for p, r in zip(fine, res)})
    best = max(solved, key=lambda p: solved[p]["profit"])
    if not np.isfinite(solved[best]["profit"]):
        return None
    return {"p": best, "s": solved[best]["s"], "profit": solved[best]["profit"], "grid": {str(p): solved[p]["profit"] for p in sorted(solved)}}


def search_b4(solver, a, w, rng, lam, gens):
    """ES over x = (p, d1, d2, d3); the (a) solution is the first candidate."""
    mean = np.array([a["p"], 0.0, 0.0, 0.0])
    scale = np.array([0.08, 0.08, 0.08, 0.08])
    sigma, mu = 1.0, max(2, lam // 4)
    wt = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wt /= wt.sum()
    best = {"x": mean.copy(), "s": a["s"], "profit": a["profit"]}
    s_hint = a["s"]
    for g in range(gens):
        X = mean + sigma * scale * rng.standard_normal((lam, 4))
        X[:, 0] = np.clip(X[:, 0], *dec.P_BOUNDS)
        X[0] = mean if g else np.array([a["p"], 0, 0, 0])
        d4 = -(X[:, 1:4] @ w[:3]) / w[3]
        D = np.c_[X[:, 1:4], d4]
        feasible = np.abs(D).max(axis=1) < 0.6
        idx = [i for i in range(lam) if feasible[i]]
        res = solver.solve([{"p": float(X[i, 0]), "d": D[i], "s0": s_hint} for i in idx])
        F = np.full(lam, -np.inf)
        for i, r in zip(idx, res):
            F[i] = r["profit"]
        order = [i for i in np.argsort(-F) if np.isfinite(F[i])]
        improved = bool(order) and F[order[0]] > best["profit"] + 1e-9
        if improved:
            k = idx.index(order[0])
            best = {"x": np.r_[X[order[0], 0], D[order[0]]], "s": res[k]["s"], "profit": float(F[order[0]])}
            s_hint = res[k]["s"]
        if order:
            top = order[:mu]
            wgt = wt[:len(top)] / wt[:len(top)].sum()
            mean = wgt @ X[top]
        sigma = float(np.clip(sigma * (1.15 if improved else 0.8), 0.15, 2.5))
        solver.log("    b4 gen {}: best search profit {:.0f} (candidates ok {})".format(g, best["profit"], int(np.isfinite(F).sum())))
    return best


def block_weights(ev, seeds):
    rows = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], seeds)[0]
    m = np.mean([r["block_online"] for r in rows], axis=0)
    return m / m.sum()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--form", choices=["hourly", "budget"], required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--opt_replicates", type=int, default=30)
    ap.add_argument("--opt_seed", type=int, default=4000)
    ap.add_argument("--final_replicates", type=int, default=20)
    ap.add_argument("--final_seed", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=480)
    ap.add_argument("--lam", type=int, default=10)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rng_seed", type=int, default=13579)
    ap.add_argument("--skip_monotone_check", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    overrides, _ = fr.setup(args)
    final_seeds = [args.final_seed + i for i in range(args.final_replicates)]
    clock = time.time()
    log = lambda m: print("{:>6.0f}s {}".format(time.time() - clock, m), flush=True)
    result = {"scenario": args.scenario, "market": overrides, "form": args.form, "method": "exact_constraint", "second_kind": "b4",
              "final_seeds": final_seeds, "opt_replicates": args.opt_replicates, "repeats": [], "monotone_check": {}}
    with mp.get_context("fork").Pool(args.workers) as pool:
        ev = dec.Evaluator(pool)
        base_final = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], final_seeds)[0]
        con_final = dec.Constraint(args.form, 1.0, base_final)
        for t in [float(v) for v in args.targets.split(",")]:
            reps = []
            for rep in range(args.repeats):
                seeds = [args.opt_seed + rep * args.opt_replicates + i for i in range(args.opt_replicates)]
                base = ev.run([np.r_[dec.BASELINE[0], dec.BASELINE[1]]], seeds)[0]
                con = dec.Constraint(args.form, t, base)
                solver = Solver(ev, con, seeds, log)
                if rep == 0 and not args.skip_monotone_check:
                    pts = [(p, np.zeros(4)) for p in (0.8, 1.0, 1.275, 1.6, 2.0)] + [(1.2, np.array(d)) for d in ([0.2, 0.0, -0.1, -0.1], [-0.2, 0.1, 0.1, 0.0], [0.3, -0.1, -0.1, -0.1])]
                    result["monotone_check"][str(t)] = solver.check_monotone(pts)
                    log("[{} W={}] G monotone in s: {}".format(args.form, t, solver.monotone))
                elif str(t) in result["monotone_check"]:
                    solver.monotone = all(r["monotone"] for r in result["monotone_check"][str(t)])
                w = block_weights(ev, seeds)
                a = search_a(solver)
                if a is None:
                    log("[{} W={}] repeat {}: no feasible (a)".format(args.form, t, rep))
                    continue
                log("[{} W={}] repeat {}: (a) p {:.3f} s {:.3f} search profit {:.0f}".format(args.form, t, rep, a["p"], a["s"], a["profit"]))
                b = search_b4(solver, a, w, np.random.default_rng(args.rng_seed + 100 * rep + int(1000 * t)), args.lam, args.gens)
                p_b, d_b = float(b["x"][0]), b["x"][1:]
                canon_a = canon_of(a["p"], np.zeros(4), a["s"])
                canon_b = canon_of(p_b, d_b, b["s"])
                ea, eb = (ev.run([c], final_seeds)[0] for c in (canon_a, canon_b))
                entry = {}
                for key, canon, rows, src in (("a", canon_a, ea, a), ("b", canon_b, eb, b)):
                    entry[key] = {"prices": canon[:4].tolist(), "shares": canon[4:].tolist(), "search_profit": src["profit"],
                                  "days": fr.day_arrays(rows, con_final)}
                entry["b"]["d"], entry["b"]["block_weights"] = d_b.tolist(), w.tolist()
                entry["a"]["price_grid"] = a["grid"]
                entry["solver_evals"] = solver.evals
                reps.append(entry)
                da = float(np.mean(entry["b"]["days"]["profit"]) - np.mean(entry["a"]["days"]["profit"]))
                log("[{} W={}] repeat {}: delta {:+.0f} | G a {:.3f} b4 {:.3f} | b4 p {:.3f} d {}".format(
                    args.form, t, rep, da, np.mean(entry["a"]["days"]["g_rel"]), np.mean(entry["b"]["days"]["g_rel"]), p_b, np.round(d_b, 3)))
                partial = dict(result, repeats=[{"target": tt, "reps": rr} for tt, rr in result.get("_done", [])] + [{"target": t, "reps": reps}])
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(dict(partial, partial=True), indent=1), encoding="utf-8")
            result.setdefault("_done", []).append((t, reps))
    result["repeats"] = [{"target": tt, "reps": rr} for tt, rr in result.pop("_done", [])]
    result["wall_seconds"] = time.time() - clock
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("done")
    return result


if __name__ == "__main__":
    main()
