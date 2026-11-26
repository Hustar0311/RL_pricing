"""v9 report: G-corrected Delta, causal (online-minutes-constrained) Delta, and the main figures.

For every v9 JSON (exact_constraint output) and repeat:
  slope  = d profit / d G of the (a) frontier from the exact (a) solutions at 0.99 W and 1.01 W (evaluation-day means)
  Delta_adj  = [profit(b4) + slope * (G_a - G_b4)] - profit(a)          (b4 moved to (a)'s realised G)
  Delta_c    = same for the constrained b4 (hourly: online minutes >= (a), budget: <= (a))
  retained share (保留比例) = Delta_c / Delta_adj: the part of the gain kept once the online-minutes restriction is imposed
  online-minutes share (可归因于在线分钟的比例) = 1 - retained share (the range is the converted retained range)
  both per repeat; median-over-median and the range over repeats
Writes summary.md / summary.json and figures (Delta_adj and online-minutes change vs W, three constraint forms).
usage: report_v9.py --v9 DIR --output_dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rep_stats(rep):
    m = lambda k, f: float(np.mean(rep[k]["days"][f]))
    out = {"profit_a": m("a", "profit"), "g_a": m("a", "g_rel"), "g_b": m("b", "g_rel"), "om_diff": m("b", "online_minutes") - m("a", "online_minutes"),
           "delta_raw": m("b", "profit") - m("a", "profit")}
    if "a_lo" in rep and "a_hi" in rep:
        dg = m("a_hi", "g_rel") - m("a_lo", "g_rel")
        out["slope"] = (m("a_hi", "profit") - m("a_lo", "profit")) / dg if abs(dg) > 1e-9 else np.nan
        out["delta_adj"] = m("b", "profit") + out["slope"] * (out["g_a"] - out["g_b"]) - out["profit_a"]
        if "bc" in rep:
            gc = m("bc", "g_rel")
            out["delta_c"] = m("bc", "profit") + out["slope"] * (out["g_a"] - gc) - out["profit_a"]
            out["om_diff_c"] = m("bc", "online_minutes") - m("a", "online_minutes")
            out["retained"] = out["delta_c"] / out["delta_adj"] if out["delta_adj"] > 0 else np.nan
    return out


def summarize(values):
    v = np.array([x for x in values if np.isfinite(x)])
    return (float(v.min()), float(np.median(v)), float(v.max())) if len(v) else (np.nan,) * 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)
    rows, summary = [], {}
    for path in sorted(Path(args.v9).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("partial") or "repeats" not in doc:
            continue
        for item in doc["repeats"]:
            stats = [rep_stats(r) for r in item["reps"]]
            keys = {k: [s[k] for s in stats if k in s] for k in stats[0]}
            entry = {"name": path.stem, "form": doc["form"], "target": item["target"], "n": len(stats)}
            for k in ("delta_raw", "delta_adj", "delta_c", "om_diff", "om_diff_c", "retained", "slope"):
                if k in keys:
                    entry[k] = summarize(keys[k])
            if "retained" in entry:
                entry["attributable"] = (1 - entry["retained"][2], 1 - entry["retained"][1], 1 - entry["retained"][0])
            entry["profit_a_median"] = float(np.median(keys["profit_a"]))
            if "delta_c" in keys and "delta_adj" in keys:
                entry["retained_median_over_median"] = float(np.median(keys["delta_c"]) / np.median(keys["delta_adj"])) if np.median(keys["delta_adj"]) > 0 else float("nan")
                entry["attributable_median_over_median"] = 1 - entry["retained_median_over_median"]
            summary[path.stem] = entry
            rows.append(entry)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    f3 = lambda t: "{:+.0f} / {:+.0f} / {:+.0f}".format(*t)
    lines = ["# v9 汇总", "", "Δ 单位为利润；每格为 最小 / 中位 / 最大（5 次重复，20 个评估日）。Δ_adj 把 (b4) 折算到 (a) 实际 G；Δ_c 为在线分钟受限的 (b4)（时薪：≥(a)，预算：≤(a)）。", "",
             "| 设置 | W̄ | 原始 Δ | 修正 Δ | 受限 Δ_c | 保留比例 (中位/中位；范围) | 可归因于在线分钟的比例 (中位/中位；范围) | 在线分钟 (b4)−(a) | 斜率 dΠ/dG |", "|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(rows, key=lambda e: (e["name"].split("_")[0], e["form"], e["name"])):
        share = attr = "—"
        if "retained_median_over_median" in e:
            share = "{:.2f}；{:.2f}–{:.2f}".format(e["retained_median_over_median"], e["retained"][0], e["retained"][2])
            attr = "{:.2f}；{:.2f}–{:.2f}".format(e["attributable_median_over_median"], e["attributable"][0], e["attributable"][2])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            e["name"], e["target"], f3(e["delta_raw"]), f3(e["delta_adj"]) if "delta_adj" in e else "—", f3(e["delta_c"]) if "delta_c" in e else "—", share, attr,
            f3(e["om_diff"]), "{:.0f}".format(e["slope"][1]) if "slope" in e else "—"))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figures(rows, out / "figures")


def figures(rows, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    forms = [("hourly", "hourly wage", "#4C78A8"), ("budget", "driver-income budget", "#F58518"), ("per_trip", "per-trip pay floor", "#54A24B")]
    for key, ylabel, name in (("delta_adj", "G-corrected $\\Delta$ = profit(b4) − profit(a)", "delta_vs_W"), ("om_diff", "online minutes: (b4) − (a)", "online_minutes_vs_W")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for form, label, color in forms:
            pts = sorted((e["target"], e[key]) for e in rows if e["form"] == form and e["name"].startswith("main") and key in e)
            if not pts:
                continue
            x = np.array([p[0] for p in pts])
            lo, med, hi = (np.array([p[1][i] for p in pts]) for i in range(3))
            ax.plot(x, med, "-o", color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("driver-metric level $\\bar W$ (relative to baseline)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / (name + ".png"), dpi=160)
        fig.savefig(fig_dir / (name + ".pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
