"""Welfare table of a setting: (a), (b4) and the online-minutes-constrained (b4), over all repeats.

Re-simulates (no new search) the stored solutions of every repeat on the evaluation days (seeds of the JSON) and reports,
as min / median / max over the repeats: platform profit, passenger (consumer) surplus - whole day and per six-hour block -,
driver net income, driver hourly wage, served orders (service quality: mean wait, timed-out leaving passengers).
The markdown has two parts: passenger surplus, and service quality.
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


FIELDS = [("profit", "平台利润", 0), ("driver_income", "司机净收入", 0), ("hourly_wage", "司机时薪", 1), ("online_minutes", "在线分钟", 0), ("served", "服务单量", 0)]
QUALITY = [("mean_wait", "平均等待(分钟)", 2), ("left", "超时离开人数", 0)]


def flat(m):
    out = {k: m[k] for k in ("profit", "consumer_surplus", "driver_income", "hourly_wage", "online_minutes", "served", "left", "mean_wait")}
    out.update({"cs_" + BLOCKS[i]: m["cs_blocks"][i] for i in range(4)})
    return out


def mmm(values, nd):
    v = np.array(values, dtype=float)
    return "{0:.{n}f} / {1:.{n}f} / {2:.{n}f}".format(v.min(), np.median(v), v.max(), n=nd)


def table(rows, names, per, label_decimals):
    lines = ["| 指标 | " + " | ".join(n for _, n in names) + " |", "|---|" + "---|" * len(names)]
    for key, label, nd in label_decimals:
        lines.append("| {} | ".format(label) + " | ".join(mmm([r[key] for r in per[k]], nd) for k, _ in names) + " |")
    return lines


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
    lines = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for item in doc["repeats"]:
            per = {}
            for key in ("a", "b", "bc"):
                if key in item["reps"][0]:
                    per[key] = []
                    for rep in item["reps"]:
                        canon = np.r_[rep[key]["prices"], rep[key]["shares"]]
                        per[key].append(flat(metrics(pool.map(day_trace, [(canon, s) for s in seeds]))))
            out["targets"][str(item["target"])] = {"n_repeats": len(item["reps"]), "per_repeat": per}
            names = [(k, n) for k, n in (("a", "(a)"), ("b", "(b4)"), ("bc", "(b4) 受限")) if k in per]
            lines += ["", "### {} W̄={}（{} 次重复，每格为 最小 / 中位 / 最大）".format(args.label, item["target"], len(item["reps"])), "",
                      "#### 平台与司机", ""] + table(None, names, per, FIELDS)
            lines += ["", "#### 乘客剩余", ""] + table(None, names, per, [("consumer_surplus", "乘客剩余（全天）", 0)] + [("cs_" + b, "乘客剩余 " + b, 0) for b in BLOCKS])
            lines += ["", "#### 服务质量", ""] + table(None, names, per, QUALITY)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / (args.label + ".json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    (Path(args.output_dir) / (args.label + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
