"""Shift the drivers' reservation incomes so that --driver_signal net reproduces the gross-signal online rate.

The net signal is lower than the gross one by the idle-cost term, so with unchanged reservation incomes fewer drivers
are online.  This tool finds the constant c such that, with every reservation income vp - c, the mean online rate at the
reference policy (p = 1, s = 0.75) under driver_signal=net equals the rate of the original scenario under gross.
Writes the shifted scenario to <workspace>/scenarios/<name>/ (git-ignored) and prints the shift.

  python tools/calibrate_net_signal.py --scenario ../scenarios/supply_binding --name supply_binding_net \
      --market passenger_wait_sensitivity=0.5,idle_cost_bearer=driver
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402
from tools.static_decoupling import MARKET  # noqa: E402

_CTX = {}


def online_rate(scenario, market, seed, warmup, price=1.0, share=0.75):
    policy = lib.constant_policy(scenario, price, share)
    state, held, _ = lib.build_start_state(scenario, policy, market, 0, seed, warmup_minutes=warmup)
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    trace = lib.simulate(env, policy, scenario.dd_test, 0, len(scenario.dd_test), market, seed, None, held)
    return float(trace[:, lib.METRICS.index("online_vehicles")].mean() / len(scenario.vp))


def _job(job):
    shift, seed = job
    scenario = copy.copy(_CTX["scenario"])
    scenario.vp = _CTX["scenario"].vp - shift
    return online_rate(scenario, _CTX["market_net"], seed, _CTX["warmup"])


def mean_rate(pool, shift, seeds):
    return float(np.mean(pool.map(_job, [(shift, s) for s in seeds])))


def fit_shift(shifts, rates, target):
    """Shift at which the least-squares line through (shift, mean online rate) reaches the target rate.

    The rate is noisy at the 0.002 level, so a line through a grid of shifts is used instead of a bisection.
    """
    slope, intercept = np.polyfit(np.asarray(shifts, dtype=float), np.asarray(rates, dtype=float), 1)
    if slope <= 0:
        raise ValueError("online rate is not increasing in the reservation shift (slope {:.3g})".format(slope))
    return float((target - intercept) / slope), float(slope)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--market", default="")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--seed0", type=int, default=7000)
    parser.add_argument("--warmup", type=int, default=480)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output_root", default=str(Path(__file__).resolve().parents[2] / "scenarios"))
    args = parser.parse_args(argv)
    overrides = dict(MARKET)
    for item in filter(None, args.market.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except ValueError:
            overrides[key] = value
    scenario = lib.Scenario(args.scenario)
    market_gross = lib.market_namespace({}, {**overrides, "driver_signal": "gross"})
    market_net = lib.market_namespace({}, {**overrides, "driver_signal": "net"})
    _CTX.update(scenario=scenario, market_net=market_net, warmup=args.warmup)
    seeds = [args.seed0 + i for i in range(args.seeds)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        target = float(np.mean(pool.starmap(online_rate, [(scenario, market_gross, s, args.warmup) for s in seeds])))
        unshifted = mean_rate(pool, 0.0, seeds)
        grid = [float(x) * float(scenario.vp.mean()) for x in (-0.1, -0.05, 0.0, 0.05, 0.1)]
        rates = [mean_rate(pool, c, seeds) for c in grid]
        shift, slope = fit_shift(grid, rates, target)
        achieved = mean_rate(pool, shift, seeds)
    print("gross online rate {:.4f}; net online rate without shift {:.4f}; grid rates {}; shift {:.4f} (slope {:.4f}/unit) -> {:.4f}".format(
        target, unshifted, np.round(rates, 4).tolist(), shift, slope, achieved))
    out = Path(args.output_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(args.scenario).glob("*.npy"):
        shutil.copy(f, out / f.name)
    np.save(out / "vp.npy", scenario.vp - shift)
    (out / "scenario.json").write_text(json.dumps({"source": args.scenario, "reservation_shift": shift, "gross_online_rate": target,
                                                    "net_online_rate_unshifted": unshifted, "net_online_rate_shifted": achieved, "grid_rates": rates, "slope_per_unit": slope, "seeds": seeds,
                                                    "market": overrides}, indent=2), encoding="utf-8")
    return shift


if __name__ == "__main__":
    main()
