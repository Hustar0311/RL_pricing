"""Counterfactual evaluation of a trained policy under a chosen market, from a simulator snapshot.

Each ``--intervention`` is compared with an untreated control on N paired common-random-number replicates
(``Environment.reseed_streams``), and reported at 1, 5, 15, 30 and 60 minutes after the fork as
control, treated, paired difference and 95% t-interval for: served orders (matched), net platform profit,
driver income signal (the 30-minute income average that drives online decisions), online vehicles, and more.

Interventions (``name:value[@minutes]``; minutes defaults to --horizon):
  price_multiply:1.1   multiply posted prices (driver pay q follows p at the policy's share unless the mode is joint)
  price_set:1.2        set posted prices
  share_set:0.85       override the driver share s (q = s p / delta) for the window; share_shift:+0.05 adds to it
  comp_multiply:1.05   scale driver pay q only
  demand_shock:0.3     entering demand x (1 + xi) for the window
  remove_drivers:3     take k drivers out of the market permanently
Use --zones 0,1 to restrict price/share/comp interventions to some origin zones.

Fork state: --snapshot (main.py --snapshot_steps ... --env_rng isolated) or --scenario + --start_minute.
Market rules default to the checkpoint's and can be overridden (--passenger_br/--driver_br/--market k=v,...).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rollout_lib as lib  # noqa: E402
from tools.measure_elasticity import lib_overrides  # noqa: E402

HORIZONS = [1, 5, 15, 30, 60]
REPORTED = ["served", "net_profit", "driver_income_signal", "online_vehicles", "revenue", "driver_total_income",
            "consumer_surplus"]


def evaluate(scenario, policy, market, state, held, t0, interventions, replicates, seed0, frequency, horizons=HORIZONS):
    env = lib.make_environment(scenario, market, seed0, frequency)
    span = max(horizons)
    if t0 + span > len(scenario.dd_test):
        raise ValueError("fork minute {} + {} exceeds the test horizon".format(t0, span))
    control = {h: [] for h in horizons}
    treated = {i.label(): {h: [] for h in horizons} for i in interventions}
    for r in range(replicates):
        seed = seed0 + r
        env.restore_state(state)
        base = lib.horizon_values(lib.simulate(env, policy, scenario.dd_test, t0, span, market, seed, None, held), horizons)
        for h in horizons:
            control[h].append(base[h])
        for arm in interventions:
            env.restore_state(state)
            values = lib.horizon_values(lib.simulate(env, policy, scenario.dd_test, t0, span, market, seed, arm, held), horizons)
            for h in horizons:
                treated[arm.label()][h].append(values[h])
    report = {}
    for label, per_h in treated.items():
        report[label] = {}
        for h in horizons:
            cell = {}
            for name in REPORTED:
                c = np.array([row[name] for row in control[h]])
                t = np.array([row[name] for row in per_h[h]])
                mean, low, high = lib.confidence_interval(t - c)
                cell[name] = {"control": float(c.mean()), "treated": float(t.mean()), "diff": mean, "ci95": [low, high]}
            report[label][str(h)] = cell
    return report


def format_table(report, names=("served", "net_profit", "driver_income_signal", "online_vehicles")):
    lines = []
    for label, per_h in report.items():
        lines.append("== {}".format(label))
        lines.append("{:>5} ".format("min") + " ".join("{:>34}".format(n) for n in names))
        for h, cell in per_h.items():
            parts = []
            for n in names:
                c = cell[n]
                parts.append("{:>9.3f}->{:<9.3f} d={:+8.3f}[{:+.3f},{:+.3f}]".format(
                    c["control"], c["treated"], c["diff"], *c["ci95"]))
            lines.append("{:>5} ".format(h) + " ".join(parts))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--constant_policy", nargs=2, type=float, metavar=("PRICE", "SHARE"))
    parser.add_argument("--snapshot")
    parser.add_argument("--start_minute", type=int, default=360)
    parser.add_argument("--intervention", action="append", required=True)
    parser.add_argument("--horizon", type=int, default=30, help="default duration of an override, in minutes")
    parser.add_argument("--zones", help="comma-separated origin zones for price/share/comp interventions")
    parser.add_argument("--replicates", "-N", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--passenger_br", choices=["on", "off", "checkpoint"], default="checkpoint")
    parser.add_argument("--driver_br", choices=["on", "off", "checkpoint"], default="checkpoint")
    parser.add_argument("--market", default="")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    scenario = lib.Scenario(args.scenario)
    if args.constant_policy:
        policy, configuration = lib.constant_policy(scenario, *args.constant_policy), {}
    else:
        policy, contract = lib.load_policy(args.checkpoint_dir, args.epoch, scenario)
        configuration = contract["configuration"]
    market = lib.market_namespace(configuration, lib_overrides(args))
    zones = [int(z) for z in args.zones.split(",")] if args.zones else None
    interventions = [lib.Intervention.parse(text, args.horizon, zones) for text in args.intervention]
    if args.snapshot:
        state, held, t0, _ = lib.load_snapshot(args.snapshot)
    else:
        state, held, t0 = lib.build_start_state(scenario, policy, market, args.start_minute, args.seed - 1)
    report = evaluate(scenario, policy, market, state, held, t0, interventions, args.replicates, args.seed, policy.frequency)
    print(format_table(report))
    if args.output:
        Path(args.output).write_text(json.dumps({"fork_minute": t0, "replicates": args.replicates, "report": report},
                                                indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
