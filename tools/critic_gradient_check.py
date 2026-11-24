"""Does the critic's price gradient point the way the simulator says?

For a joint-runner checkpoint and N test-day snapshots (the market state reached by following the policy to minute t0
after a warm-up) the script computes
  * the critic gradient along the uniform price direction, dQ1/d ln p = sum_i dQ1/d a_i / ln 3 (a_i: normalised log-price
    action of zone i, price = 3^a_i), at the actor's own action; and the finite difference Q1(a+h) - Q1(a-h), h = ln(1.05)/ln 3;
  * the simulated counterfactual: all prices x1.05 versus x0.95 for the next H minutes (default 120) from the same
    snapshot, R paired common-random-number replicates, discounted profit return with gamma per minute (default 0.99).
It reports how often the critic's sign equals the simulator's sign (all snapshots, and only snapshots whose paired
counterfactual difference is significant at 95%).
usage: critic_gradient_check.py --scenario S --market k=v,... --dir D --epoch E --output O
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import frontier_decoupling as fr  # noqa: E402
from tools import static_decoupling as dec  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402

STEP = 0.05


def snapshot_job(job):
    directory, epoch, t0, seed, horizon, replicates, gamma = job
    torch.set_num_threads(1)
    scenario, market = dec._CTX["scenario"], dec._CTX["market"]
    policy = lib.load_policy(directory, epoch, scenario)[0]
    controller = policy.controller
    state, held, _ = lib.build_start_state(scenario, policy, market, t0, seed, warmup_minutes=dec._CTX["warmup"])
    env = lib.make_environment(scenario, market, seed, policy.frequency)
    env.restore_state(state)
    s, s2d = controller.observe(env, policy.od_permutation)
    t = torch.tensor([[t0]])
    h = math.log(1 + STEP) / math.log(3.0)
    with torch.no_grad():
        action = controller.pricer.actor(s[None], s2d[None], t)
    a = action.clone().requires_grad_(True)
    q1 = controller.pricer.critic(s[None], s2d[None], a, t)[0]
    grad = torch.autograd.grad(q1.sum(), a)[0]
    with torch.no_grad():
        up = controller.pricer.critic(s[None], s2d[None], action + h, t)[0].item()
        down = controller.pricer.critic(s[None], s2d[None], action - h, t)[0].item()
    weights = gamma ** np.arange(horizon)
    col = lib.METRICS.index("net_profit")
    values = {"control": [], "up": [], "down": []}
    for r in range(replicates):
        for name, arm in (("control", None), ("up", lib.Intervention("price_multiply", 1 + STEP, horizon)),
                          ("down", lib.Intervention("price_multiply", 1 - STEP, horizon))):
            env.restore_state(state)
            trace = lib.simulate(env, policy, scenario.dd_test, t0, horizon, market, 9000 + 31 * t0 + r, arm, held)
            values[name].append(float((trace[:, col] * weights).sum()))
    diff = np.array(values["up"]) - np.array(values["down"])
    mean, low, high = lib.confidence_interval(diff)
    return {"t0": t0, "dq_dlogp": float(grad.sum()) / math.log(3.0), "q_fd": up - down, "return_diff": mean,
            "ci95": [low, high], "up_vs_control": float(np.mean(values["up"]) - np.mean(values["control"])),
            "down_vs_control": float(np.mean(values["down"]) - np.mean(values["control"])),
            "mean_action": float(action.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--snapshots", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    args.warmup = 1440
    fr.setup(args)
    span = len(dec._CTX["scenario"].dd_test) - args.horizon - 30
    starts = [int(30 + i * span / max(args.snapshots - 1, 1)) for i in range(args.snapshots)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(snapshot_job, [(args.dir, args.epoch, t0, args.seed, args.horizon, args.replicates, args.gamma) for t0 in starts])
    sign = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    agree = [sign(r["dq_dlogp"]) == sign(r["return_diff"]) for r in rows]
    agree_fd = [sign(r["q_fd"]) == sign(r["return_diff"]) for r in rows]
    sig = [not (r["ci95"][0] <= 0 <= r["ci95"][1]) for r in rows]
    result = {"dir": args.dir, "epoch": args.epoch, "horizon": args.horizon, "gamma": args.gamma, "replicates": args.replicates,
              "snapshots": rows, "agree_gradient": float(np.mean(agree)), "agree_finite_difference": float(np.mean(agree_fd)),
              "n_significant": int(sum(sig)),
              "agree_gradient_significant": float(np.mean([a for a, s in zip(agree, sig) if s])) if any(sig) else None,
              "share_critic_up": float(np.mean([r["dq_dlogp"] > 0 for r in rows])),
              "share_sim_up": float(np.mean([r["return_diff"] > 0 for r in rows]))}
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print({k: v for k, v in result.items() if k != "snapshots"})
    return result


if __name__ == "__main__":
    main()
