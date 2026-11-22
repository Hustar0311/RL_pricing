"""Collect the v2 static-decoupling runs into the main tables (markdown on stdout).

Main table: constraint form x level x scenario (eps_S); Delta' = (b') - (a'), its constraint-adjusted version, the
allocation / efficiency split, change in online driver minutes and the supply-channel term, constraint satisfaction.
Source test: Delta' is "driven by fewer online driver minutes" when Delta' (or its adjusted version) is positive with
a 95% interval above 0, the online-minute change is significantly negative and the supply-channel term is more than
half of Delta'.
"""
import argparse
import glob
import json
import re
from pathlib import Path

EPS = {"low": 0.076, "mid": 0.394, "high": 0.531}


def fmt(cell, digits=0):
    return "{:+.{d}f} [{:+.{d}f}, {:+.{d}f}]".format(cell["mean"], cell["lo"], cell["hi"], d=digits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "decouple-benchmark" / "v2"))
    parser.add_argument("--elasticities", default=None, help="json {low,mid,high: eps_S}; defaults to measured values")
    args = parser.parse_args(argv)
    eps = dict(EPS)
    for name in EPS:
        path = Path(args.dir) / "elasticity_{}.json".format(name)
        if path.exists():
            eps[name] = json.loads(path.read_text())["epsilon_S_tail_mean"]["elasticity_mean"]
    rows = []
    for path in sorted(glob.glob(str(Path(args.dir) / "*.json"))):
        if "elasticity_" in path:
            continue
        r = json.loads(Path(path).read_text())
        name = Path(path).stem
        m = re.match(r"(low|mid|high)_(\w+?)_([\d.]+)(?:_(payweighted|bigbudget))?$", name)
        if not m:
            continue
        rows.append((m.group(1), r["constraint"], r["level"], m.group(4) or "", r, name))
    order = {"low": 0, "mid": 1, "high": 2}
    rows.sort(key=lambda x: (x[3], ["hourly", "budget", "per_trip"].index(x[1]), order[x[0]], x[2]))
    print("| 设置 | 形式 | 水平 | ε_S | Δ'=(b')−(a') | 约束调整后Δ' | 分配 | 效率 | Δ在线分钟 | 供给渠道项 | (a')/(b') 日满足率 | (b')分成贴边界 | Δ=(b)−(a) |")
    print("|---|---|---:|---:|---|---|---:|---:|---|---:|---|---|---|")
    summary = []
    for scen, form, level, tag, r, name in rows:
        d = r["deltas"]["b1_vs_a1"]
        a1, b1 = r["policies"]["a1"], r["policies"]["b1"]
        main_cell, adj = d["profit"], d.get("profit_constraint_adjusted", d["profit"])
        minutes = d["online_minutes"]
        supply = d["supply_channel"]["mean"]
        share = supply / main_cell["mean"] if abs(main_cell["mean"]) > 1e-9 else float("nan")
        significant_gain = (main_cell["lo"] > 0) or (adj["lo"] > 0)
        driven = bool(significant_gain and minutes["hi"] < 0 and share > 0.5)
        summary.append((name, driven, main_cell, adj, minutes, supply, share))
        print("| {} | {} | {:g} | {:.2f} | {} | {} | {:+.0f} | {:+.0f} | {} | {:+.0f} | {:.2f}/{:.2f} | {} | {} |".format(
            name, form, level, eps[scen], fmt(main_cell), fmt(adj), d["allocation"]["mean"], d["efficiency"]["mean"],
            fmt(minutes), supply, a1["constraint_day_satisfaction"], b1["constraint_day_satisfaction"],
            "是" if b1["shares_at_bound"] else "否", fmt(r["deltas"]["b_vs_a"]["profit"])))
    print("\n在线司机分钟数下降是 Δ' 主要来源的设置：")
    for name, driven, main_cell, adj, minutes, supply, share in summary:
        if driven:
            print("- {}: Δ'={:+.0f}, Δ在线分钟={:+.0f}, 供给渠道项={:+.0f} ({:.0%} of Δ')".format(name, main_cell["mean"], minutes["mean"], supply, share))
    return summary


if __name__ == "__main__":
    main()
