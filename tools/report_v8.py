"""v8 report: exact-constraint solver vs the v5 (b) and v7 (b4) penalty searches.

Merges the per-target exact_constraint JSONs into one file per constraint form (input of mechanism_plots.py) and writes a
table of Delta = profit(second) - profit(a) on the 20 evaluation days over 5 independent repeats (min / median / max /
range, plus the range of (a)'s own profit and the mean G/G_base on the evaluation days).
usage: report_v8.py --v5 D --v7 D --v8 D --output summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(item):
    d = np.array([np.mean(r["b"]["days"]["profit"]) - np.mean(r["a"]["days"]["profit"]) for r in item["reps"]])
    a = np.array([np.mean(r["a"]["days"]["profit"]) for r in item["reps"]])
    g = np.mean([np.mean(r["a"]["days"]["g_rel"]) for r in item["reps"]]), np.mean([np.mean(r["b"]["days"]["g_rel"]) for r in item["reps"]])
    return d, a, g


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("v5", "v7", "v8", "output"):
        ap.add_argument("--" + k, required=True)
    args = ap.parse_args(argv)
    lines = ["# v8 求解方法对比：精确约束 vs ε 约束罚函数", "",
             "Δ = 利润(第二类) − 利润(a)，20 个评估日，5 次独立重复。(b)：价格+8 分成，罚函数，搜索 10 天（v5）；(b4)：统一价+4 时段分成，罚函数，30 天（v7）；(b4 精确)：精确约束求解，30 天（v8）。",
             "", "| 约束 | W̄ | 方法 | Δ 最小 | Δ 中位 | Δ 最大 | Δ 极差 | (a) 利润极差 | G/G_base (a, 第二类) |", "|---|---|---|---|---|---|---|---|---|"]
    for form in ("hourly", "budget"):
        sources = [("(b) 罚函数", load(Path(args.v5) / "rep_low_platform_gross_{}.json".format(form))),
                   ("(b4) 罚函数", load(Path(args.v7) / "rep_b4_low_platform_gross_{}.json".format(form)))]
        parts = [load(p) for p in sorted(Path(args.v8).glob("exact_low_platform_gross_{}_*.json".format(form)))]
        merged = dict(parts[0], repeats=[r for p in parts for r in p["repeats"]])
        merged["repeats"].sort(key=lambda r: r["target"])
        Path(args.v8, "exact_low_platform_gross_{}.json".format(form)).write_text(json.dumps(merged), encoding="utf-8")
        sources.append(("(b4) 精确约束", merged))
        for item in sources[0][1]["repeats"]:
            t = item["target"]
            for label, doc in sources:
                it = next(x for x in doc["repeats"] if x["target"] == t)
                d, a, g = stats(it)
                lines.append("| {} | {:.2f} | {} | {:+.0f} | {:+.0f} | {:+.0f} | {:.0f} | {:.0f} | {:.3f}, {:.3f} |".format(
                    form, t, label, d.min(), np.median(d), d.max(), d.max() - d.min(), a.max() - a.min(), *g))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
