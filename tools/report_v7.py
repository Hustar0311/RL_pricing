"""v7 noise comparison: Delta of (b4) [30 search days] against the v5 (b) [10 search days] from 5 independent searches.

Delta = profit(second class) - profit(a) on the 20 evaluation days, per repeat.  Reports min / median / max and the range
(max - min) of Delta for both classes, plus the spread of (a)'s own evaluated profit across repeats.
usage: report_v7.py --v5 DIR --v7 DIR --output summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

SETTINGS = [("hourly", "rep_low_platform_gross_hourly", "rep_b4_low_platform_gross_hourly"),
            ("budget", "rep_low_platform_gross_budget", "rep_b4_low_platform_gross_budget")]


def deltas(item):
    return np.array([np.mean(r["b"]["days"]["profit"]) - np.mean(r["a"]["days"]["profit"]) for r in item["reps"]])


def a_profits(item):
    return np.array([np.mean(r["a"]["days"]["profit"]) for r in item["reps"]])


def fmt(x):
    return "{:+.0f}".format(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True)
    ap.add_argument("--v7", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    lines = ["# v7 噪声对比：(b4) 对 (b)", "",
             "Δ = 利润(第二类策略) − 利润(a)，20 个评估日，5 次独立 ε 约束搜索。(b)：价格+8 分成，搜索 10 天（v5/v6）；(b4)：统一价+4 时段分成，搜索 30 天（v7，(a) 也重新用 30 天搜索）。",
             "(b4)⊃(a)、(b)⊃(b4)；评估用新日，Δ 是下界，负值读作检测不到增益。", "",
             "| 约束 | W̄ | 类 | 搜索日 | Δ 最小 | Δ 中位 | Δ 最大 | 极差 | (a) 利润极差 |", "|---|---|---|---|---|---|---|---|---|"]
    for form, old, new in SETTINGS:
        p5, p7 = Path(args.v5) / (old + ".json"), Path(args.v7) / (new + ".json")
        if not (p5.exists() and p7.exists()):
            continue
        d5, d7 = json.loads(p5.read_text(encoding="utf-8")), json.loads(p7.read_text(encoding="utf-8"))
        for i5 in d5["repeats"]:
            i7 = next((x for x in d7["repeats"] if x["target"] == i5["target"]), None)
            if i7 is None:
                continue
            for label, item, days in (("(b)", i5, len(d5["opt_seeds"])), ("(b4)", i7, len(d7["opt_seeds"]))):
                d, a = deltas(item), a_profits(item)
                lines.append("| {} | {:.2f} | {} | {} | {} | {} | {} | {:.0f} | {:.0f} |".format(
                    form, i5["target"], label, days, fmt(d.min()), fmt(np.median(d)), fmt(d.max()), d.max() - d.min(), a.max() - a.min()))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
