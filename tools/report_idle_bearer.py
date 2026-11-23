"""Tables for the idle-cost-bearer study (v3): platform vs driver bearer, background contrasts, seed robustness.

Main metric is the constraint-adjusted Delta' = (b') - (a') (``profit_constraint_adjusted``); the unadjusted value is
listed beside it.  Background contrasts: (a')-(a) (price flexibility) and (a)-(f) (decoupling-free share optimisation,
f = best price at the fixed share 0.75), each with its paired 95% interval over the 20 final days.
"""
import argparse
import glob
import json
from pathlib import Path


def cell(c, digits=0):
    return "{:+.{d}f} [{:+.{d}f}, {:+.{d}f}]".format(c["mean"], c["lo"], c["hi"], d=digits)


def sign_word(c):
    return "显著>0" if c["lo"] > 0 else ("显著<0" if c["hi"] < 0 else "含0")


def load(directory):
    runs = {}
    for path in sorted(glob.glob(str(Path(directory) / "*.json"))):
        runs[Path(path).stem] = json.loads(Path(path).read_text())
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "decouple-benchmark" / "v3"))
    args = parser.parse_args(argv)
    runs = load(args.dir)
    main_runs = {k: r for k, r in runs.items() if k.startswith("low_")}
    print("### 主表（low 场景，ε_S≈0.08）：主指标为调整后 Δ′\n")
    print("| 约束 | 水平 | 成本承担方 | 调整后Δ′ | 未调整Δ′ | 分配 | 效率 | Δ在线分钟 | 供给渠道项 | (a′)/(b′)日满足率 | κ |")
    print("|---|---:|---|---|---|---:|---:|---|---:|---|---:|")
    for name, r in main_runs.items():
        form, level, bearer = r["constraint"], r["level"], r["market"].get("idle_cost_bearer", "platform")
        d = r["deltas"]["b1_vs_a1"]
        adj = d.get("profit_constraint_adjusted", d["profit"])
        print("| {} | {:g} | {} | {} | {} | {:+.0f} | {:+.0f} | {} | {:+.0f} | {:.2f}/{:.2f} | {:.3g} |".format(
            form, level, bearer, cell(adj), cell(d["profit"]), d["allocation"]["mean"], d["efficiency"]["mean"],
            cell(d["online_minutes"]), d["supply_channel"]["mean"], r["policies"]["a1"]["constraint_day_satisfaction"],
            r["policies"]["b1"]["constraint_day_satisfaction"], r["kappa_profit_per_relative_slack"]))
    print("\n### 在线分钟数是否仍显著为负（per_trip 与 hourly）\n")
    print("| 约束 | 水平 | 承担方 | Δ在线分钟 | 判定 | 供给渠道项/未调整Δ′ |")
    print("|---|---:|---|---|---|---:|")
    for name, r in main_runs.items():
        if r["constraint"] not in ("per_trip", "hourly"):
            continue
        d = r["deltas"]["b1_vs_a1"]
        share = d["supply_channel"]["mean"] / d["profit"]["mean"] if abs(d["profit"]["mean"]) > 1e-9 else float("nan")
        print("| {} | {:g} | {} | {} | {} | {:.0%} |".format(r["constraint"], r["level"], r["market"].get("idle_cost_bearer", "platform"),
                                                       cell(d["online_minutes"]), sign_word(d["online_minutes"]), share))
    print("\n### 背景对照（配对 95% 区间；括号内为约束调整后）\n")
    print("| 约束 | 水平 | 承担方 | (a′)−(a) | 调整后 | (a)−(f) | 调整后 |")
    print("|---|---:|---|---|---|---|---|")
    for name, r in main_runs.items():
        bearer = r["market"].get("idle_cost_bearer", "platform")
        c1, c2 = r["deltas"]["a1_vs_a"], r["deltas"]["a_vs_f"]
        print("| {} | {:g} | {} | {} | {} | {} | {} |".format(r["constraint"], r["level"], bearer, cell(c1["profit"]),
                                                          cell(c1.get("profit_constraint_adjusted", c1["profit"])), cell(c2["profit"]),
                                                          cell(c2.get("profit_constraint_adjusted", c2["profit"]))))
    seeds = {k: r for k, r in runs.items() if k.startswith("seed")}
    if seeds:
        print("\n### 场景种子稳健性（platform）：调整后 Δ′\n")
        print("| 约束 | 场景种子 | 调整后Δ′ | 符号 | 未调整Δ′ | Δ在线分钟 |")
        print("|---|---:|---|---|---|---|")
        rows = [(r["constraint"], 0, r) for k, r in main_runs.items()
                if r["level"] == 1.0 and r["constraint"] in ("hourly", "budget") and r["market"].get("idle_cost_bearer", "platform") == "platform"]
        rows += [(r["constraint"], int(k[4]), r) for k, r in seeds.items()]
        for form, seed, r in sorted(rows, key=lambda x: (x[0], x[1])):
            d = r["deltas"]["b1_vs_a1"]
            adj = d.get("profit_constraint_adjusted", d["profit"])
            print("| {} | {} | {} | {} | {} | {} |".format(form, seed, cell(adj), sign_word(adj), cell(d["profit"]), cell(d["online_minutes"])))


if __name__ == "__main__":
    main()
