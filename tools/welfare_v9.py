"""Welfare table of a setting: (a), (b4) and the online-minutes-constrained (b4) of the median-Delta repeat.

Re-simulates the three stored solutions of the repeat whose raw Delta = profit(b4) - profit(a) is the median over repeats
on the evaluation days (seeds of the JSON) and reports platform profit, passenger (consumer) surplus - whole day and per
six-hour block -, driver net income, driver hourly wage, served orders, timed-out leaving passengers and mean wait.
usage: welfare_v9.py --scenario S --market k=v,... --json J --label L --output_dir D
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import frontier_decoupling as fr  # noqa: E402
from tools import static_decoupling as dec  # noqa: E402
from tools import rollout_lib as lib  # noqa: E402
from tools.mechanism_plots import day_trace  # noqa: E402

BLOCKS = ["0-6h", "6-12h", "12-18h", "18-24h"]


def metrics(traces):
    col = lib.METRICS.index
    tot = lambda name: float(np.mean([t[:, col(name)].sum() for t in traces]))
    blk = lambda name: [float(np.mean([t[b * 360:(b + 1) * 360, col(name)].sum() for t in traces])) for b in range(4)]
    online = tot("online_vehicles")
    income = tot("driver_total_income") - tot("driver_idle_cost")
    served = tot("served")
    return {"profit": tot("net_profit"), "consumer_surplus": tot("consumer_surplus"), "cs_blocks": blk("consumer_surplus"),
            "driver_income": income, "hourly_wage": income / online * 60.0, "served": served, "left": tot("left"),
            "mean_wait": tot("wait_minutes") / served, "online_minutes": online, "profit_blocks": blk("net_profit")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--market", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    args.warmup = 480
    fr.setup(args)
    doc = json.loads(Path(args.json).read_text(encoding="utf-8"))
    seeds = doc["final_seeds"]
    out = {"label": args.label, "targets": {}}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for item in doc["repeats"]:
            deltas = [np.mean(r["b"]["days"]["profit"]) - np.mean(r["a"]["days"]["profit"]) for r in item["reps"]]
            idx = int(np.argsort(deltas)[(len(deltas) - 1) // 2])
            rep = item["reps"][idx]
            entry = {"repeat": idx, "delta": deltas[idx]}
            for key in ("a", "b", "bc"):
                if key in rep:
                    canon = np.r_[rep[key]["prices"], rep[key]["shares"]]
                    entry[key] = metrics(pool.map(day_trace, [(canon, s) for s in seeds]))
            out["targets"][str(item["target"])] = entry
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / (args.label + ".json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = []
    for t, e in out["targets"].items():
        names = [(k, n) for k, n in (("a", "(a)"), ("b", "(b4)"), ("bc", "(b4) 受限")) if k in e]
        lines += ["", "### {} W̄={}（第 {} 次重复，Δ={:+.0f}）".format(args.label, t, e["repeat"], e["delta"]), "",
                  "| 指标 | " + " | ".join(n for _, n in names) + " |", "|---|" + "---|" * len(names)]
        rows = [("平台利润", lambda m: m["profit"]), ("乘客剩余（全天）", lambda m: m["consumer_surplus"])]
        rows += [("乘客剩余 " + BLOCKS[i], (lambda i: lambda m: m["cs_blocks"][i])(i)) for i in range(4)]
        rows += [("司机净收入", lambda m: m["driver_income"]), ("司机时薪", lambda m: m["hourly_wage"]), ("在线分钟", lambda m: m["online_minutes"]),
                 ("服务单量", lambda m: m["served"]), ("超时离开人数", lambda m: m["left"]), ("平均等待(分钟)", lambda m: m["mean_wait"])]
        for label, f in rows:
            lines.append("| {} | ".format(label) + " | ".join("{:.2f}".format(f(e[k])) if "等待" in label else "{:.0f}".format(f(e[k])) for k, _ in names) + " |")
    (Path(args.output_dir) / (args.label + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
