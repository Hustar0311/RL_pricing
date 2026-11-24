"""v6/summary.md: Delta, attributable share and welfare changes by constraint x W x cost basis (draft of the paper's results table).

Delta = profit(b) - profit(a) at the same driver-metric level W; (b_c) = (b) re-optimised under an online-minutes constraint
relative to (a) (hourly: minutes >= (a)'s, budget: minutes <= (a)'s).  Attributable share = (Delta_b - Delta_c) / Delta_b.
Where five independent searches exist (low, platform+gross: hourly 1.0/1.2, budget 0.8/1.0) the share is
median(Delta_b)-vs-median(Delta_c) and the range spans all min/median/max combinations of the two; elsewhere it comes from a
single search and no range is given.
usage: report_v6.py --v5 DIR --v6 DIR --output summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np


def mean(entry, key):
    return float(np.mean(entry["days"][key]))


def rec(a, b, c=None):
    out = {"d_b": mean(b, "profit") - mean(a, "profit"), "cs_b": mean(b, "consumer_surplus") - mean(a, "consumer_surplus"),
           "inc_b": mean(b, "income") - mean(a, "income"), "g_a": mean(a, "g_rel"), "g_b": mean(b, "g_rel")}
    if c is not None:
        out.update(d_c=mean(c, "profit") - mean(a, "profit"), cs_c=mean(c, "consumer_surplus") - mean(a, "consumer_surplus"),
                   inc_c=mean(c, "income") - mean(a, "income"), om_c=mean(c, "online_minutes") / mean(a, "online_minutes"))
    return out


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def share_range(d_b, d_c):
    stats = lambda v: [min(v), float(np.median(v)), max(v)]
    sb, sc = stats(d_b), stats(d_c)
    mid = (sb[1] - sc[1]) / sb[1] if sb[1] > 0 else float("nan")
    combos = [(b - c) / b for b in sb for c in sc if b > 0]
    return mid, (min(combos), max(combos)) if combos else (float("nan"), float("nan"))


def collect(v5, v6):
    rows = []  # (constraint, W, basis, list of per-search records)

    def add(constraint, w, basis, records):
        rows.append({"constraint": constraint, "W": w, "basis": basis, "reps": records})

    # hourly, single searches (v5); causal (floor) where available
    for name, basis in (("low_platform_gross_hourly", "platform+gross"), ("low_driver_net_hourly", "driver+net")):
        doc = load(v5 / (name + ".json"))
        for p in doc["points"]:
            add("hourly", p["target"], basis, [rec(p["a"], p["b"], p.get("b_floor"))])
    # hourly / budget with five independent searches (platform+gross)
    for form, rep_name, cons_name in (("hourly", "rep_low_platform_gross_hourly", "rep_cons_low_platform_gross_hourly"),
                                      ("budget", "rep_low_platform_gross_budget", "rep_cons_low_platform_gross_budget")):
        rep, cons = load(v5 / (rep_name + ".json")), load(v6 / (cons_name + ".json"))
        for item in rep["repeats"]:
            cons_item = None if cons is None else next((c for c in cons["repeats"] if abs(c["target"] - item["target"]) < 1e-9), None)
            records = [rec(r["a"], r["b"], None if cons_item is None else cons_item["reps"][i]["b_c"]) for i, r in enumerate(item["reps"])]
            # these positions are reported from the repeats; drop the single-search row of the same position
            rows[:] = [x for x in rows if not (x["constraint"] == form and abs(x["W"] - item["target"]) < 1e-9 and x["basis"] == "platform+gross")]
            add(form, item["target"], "platform+gross", records)
    # budget single searches with the online-minutes cap (v6)
    for name, basis in (("cap_low_platform_gross_budget", "platform+gross"), ("cap_low_driver_net_budget", "driver+net")):
        doc = load(v6 / (name + ".json"))
        if doc is None:
            continue
        for p in doc["points"]:
            if basis == "platform+gross" and any(x["constraint"] == "budget" and x["basis"] == basis and abs(x["W"] - p["target"]) < 1e-9 for x in rows):
                continue
            add("budget", p["target"], basis, [rec(p["a"], p["b"], p.get("b_cap"))])
    return sorted(rows, key=lambda x: (x["constraint"], x["basis"], x["W"]))


def fmt_range(v):
    return "{:+.0f} / {:+.0f} / {:+.0f}".format(min(v), np.median(v), max(v)) if len(v) > 1 else "{:+.0f}".format(v[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    rows = collect(Path(args.v5), Path(args.v6))
    out = ["# v6 汇总：Δ、可归因比例与福利变化（论文结果表底稿）", "",
           "Δ=利润(b)−利润(a)，(b) 包含 (a)、评估用新日，所以 Δ 是真实增益的下界；负值读作“检测不到增益”。",
           "受限 (b_c)：hourly 加“在线分钟 ≥ 同位置 (a)”，budget 加“在线分钟 ≤ 同位置 (a)”后重新优化。可归因比例＝(Δ_b−Δ_c)/Δ_b。",
           "n=5 的行（platform+gross：hourly 1.0/1.2，budget 0.8/1.0）为 5 次独立搜索：Δ 列为 最小 / 中位数 / 最大，福利变化取中位数，可归因比例为中位数对中位数，括号为 3×3 个 最小/中位/最大 组合的范围；n=1 的行没有范围。",
           "福利变化 = 对应解减去同位置 (a)：乘客剩余、司机净收入（扣除司机承担的空驶成本）、平台利润（=Δ）。",
           "",
           "| 约束 | W̄ | 口径 | n | Δ=(b)−(a) 最小/中位/最大 | 可归因比例 [范围] | 乘客剩余变化 (b) | 司机净收入变化 (b) | 平台利润变化 (b) | Δ_c=(b_c)−(a) | 乘客剩余变化 (b_c) | 司机净收入变化 (b_c) | b_c 在线分钟/(a) |",
           "|" + "---|" * 13]
    for r in rows:
        reps = r["reps"]
        d_b = [x["d_b"] for x in reps]
        has_c = all("d_c" in x for x in reps)
        if has_c:
            d_c = [x["d_c"] for x in reps]
            mid, (lo, hi) = share_range(d_b, d_c) if len(reps) > 1 else ((d_b[0] - d_c[0]) / d_b[0] if d_b[0] > 0 else float("nan"), (None, None))
            share = "n/a" if not np.isfinite(mid) else ("{:.0%}".format(mid) if lo is None else "{:.0%} [{:.0%}, {:.0%}]".format(mid, lo, hi))
        else:
            share = "n/a"
        med = lambda key: float(np.median([x[key] for x in reps]))
        c_cells = ["n/a"] * 4
        if has_c:
            c_cells = ["{:+.0f}".format(med("d_c")), "{:+.0f}".format(med("cs_c")), "{:+.0f}".format(med("inc_c")), "{:.3f}".format(med("om_c"))]
        out.append("| {} | {:.2f} | {} | {} | {} | {} | {:+.0f} | {:+.0f} | {:+.0f} | {} |".format(
            r["constraint"], r["W"], r["basis"], len(reps), fmt_range(d_b), share, med("cs_b"), med("inc_b"), med("d_b"), " | ".join(c_cells)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
