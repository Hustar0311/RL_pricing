"""Tables and frontier plots from frontier_decoupling JSON files (v4 study).

usage: python tools/report_frontier.py --input DIR [--ext DIR/ext] --output DIR
For every run the extended file (ext/<name>.json) is used when present.  Writes <output>/tables.md and one PNG per run
(profit-vs-G frontiers of (a) and (b) with the Delta(m) band and online minutes along the frontier).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import frontier_decoupling as fr  # noqa: E402

MULTS = fr.MULTS


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(res, key):
    pts = sorted(res["points"], key=lambda p: p["lambda"])
    return {k: np.array([p[k]["days"][key] for p in pts]) for k in ("a", "b")}, pts


def fmt(v, digits=0):
    return "n/a" if v is None or not np.isfinite(v) else "{:.{}f}".format(v, digits)


def cell(point, lo, hi):
    return "n/a" if not np.isfinite(point) else "{:+.0f} [{:+.0f}, {:+.0f}]".format(point, lo, hi)


def metric_delta(res, key, n_boot):
    g, _ = arrays(res, "g_rel")
    m, _ = arrays(res, key)
    return fr.bootstrap_delta(g["a"], m["a"], g["b"], m["b"], MULTS, n_boot, seed=1)


def plot(res, name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, pts = arrays(res, "g_rel")
    prof, _ = arrays(res, "profit")
    om, _ = arrays(res, "online_minutes")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for kind, color in (("a", "#1b6ca8"), ("b", "#d1495b")):
        order = np.argsort(g[kind].mean(1))
        x = g[kind].mean(1)[order]
        ax[0].plot(x, prof[kind].mean(1)[order], "o-", color=color, label="(%s) %s" % (kind, "p + s" if kind == "a" else "p + 8 shares"))
        ax[2].plot(x, om[kind].mean(1)[order], "o-", color=color, label="(%s)" % kind)
    ax[0].axvspan(0.8, 1.25, color="0.93", zorder=0)
    ax[0].set(xlabel="driver metric / reference", ylabel="platform profit", title="frontiers")
    d = res["delta_curve"]
    grid, mean, lo, hi = (np.array(d[k], dtype=float) for k in ("grid", "mean", "lo", "hi"))
    ax[1].plot(grid, mean, color="k")
    ax[1].fill_between(grid, lo, hi, color="0.8")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set(xlabel="driver metric / reference", ylabel="Delta = profit(b) - profit(a)", title="Delta(m), 95% bootstrap over days")
    ax[2].set(xlabel="driver metric / reference", ylabel="online minutes / day", title="online minutes along the frontier")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False)
    ax[2].legend(frameon=False)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in src.glob("*.json"))
    lines = ["# 前沿法结果（v4）", ""]
    summary = ["## Delta(W̄) 汇总：利润(b)−利润(a)，括号为 95% bootstrap 区间（评估日）", "",
               "| 运行 | 0.8 W̄ | 1.0 W̄ | 1.2 W̄ | 前沿点数(基础+补点) |", "|---|---|---|---|---|"]
    detail = []
    for name in names:
        ext = src / "ext" / (name + ".json")
        res = load(ext if ext.exists() else src / (name + ".json"))
        base = load(src / (name + ".json"))
        n_new = len(res["points"]) - len(base["points"])
        d = res["delta_at"]
        summary.append("| {} | {} | {} | {} | {}+{} |".format(name, *[cell(d[str(m)]["mean"], d[str(m)]["lo"], d[str(m)]["hi"]) for m in MULTS], len(base["points"]), n_new))
        plot(res, name, out / (name + ".png"))
        detail += ["", "### " + name, "", "基准 G=%.4g，基准利润 %.0f。" % (res["baseline_value"], res["baseline"]["profit"]), ""]
        detail += ["| λ | (a) G | (a) 利润 | (a) 在线分钟 | (a) 服务量 | (a) 消费者剩余 | (b) G | (b) 利润 | (b) 在线分钟 | (b) 服务量 | (b) 消费者剩余 |", "|" + "---|" * 11]
        for p in sorted(res["points"], key=lambda q: q["lambda"]):
            row = [fmt(p["lambda"])]
            for k in ("a", "b"):
                dd = p[k]["days"]
                row += [fmt(np.mean(dd["g_rel"]), 3), fmt(np.mean(dd["profit"])), fmt(np.mean(dd["online_minutes"])), fmt(np.mean(dd["served"])), fmt(np.mean(dd["consumer_surplus"]))]
            detail.append("| " + " | ".join(row) + " |" + (" 补点" if p.get("added") else ""))
        om = metric_delta(res, "online_minutes", args.bootstrap)
        sv = metric_delta(res, "served", args.bootstrap)
        cs = metric_delta(res, "consumer_surplus", args.bootstrap)
        detail += ["", "(b)−(a) 在 W̄ 处的差：", "", "| 指标 | 0.8 W̄ | 1.0 W̄ | 1.2 W̄ |", "|---|---|---|---|"]
        for label, t in (("在线分钟", om), ("服务量", sv), ("消费者剩余", cs)):
            detail.append("| {} | {} |".format(label, " | ".join(cell(t[0][i], t[1][i], t[2][i]) for i in range(3))))
        pt = res["delta_curve"]
        dm = np.array(pt["mean"], dtype=float)
        g = np.array(pt["grid"])
        ok = np.isfinite(dm)
        if ok.sum() >= 3:
            om_curve = fr.interpolate(arrays(res, "g_rel")[0]["b"].mean(1), arrays(res, "online_minutes")[0]["b"].mean(1), g) - \
                fr.interpolate(arrays(res, "g_rel")[0]["a"].mean(1), arrays(res, "online_minutes")[0]["a"].mean(1), g)
            both = ok & np.isfinite(om_curve)
            if both.sum() >= 3:
                rho = np.corrcoef(dm[both], om_curve[both])[0, 1]
                detail.append("\n在可得区间 G∈[{:.2f},{:.2f}] 上，Δ(m) 与在线分钟差(b−a) 的相关系数 {:+.2f}。".format(g[both].min(), g[both].max(), rho))
        if res.get("variance"):
            detail += ["", "优化器方差（5 次独立搜索，目标值 Π+λG/G_base）：", "", "| λ 网格位 | λ | (a) 均值 | (a) 标准差 | (a) 极差 | (b) 均值 | (b) 标准差 | (b) 极差 | (b)−(a) 均值 |", "|" + "---|" * 9]
            for idx, v in res["variance"].items():
                fa = np.array([r["objective_a"] for r in v["reps"]])
                fb = np.array([r["objective_b"] for r in v["reps"]])
                detail.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    idx, fmt(v["lambda"]), fmt(fa.mean()), fmt(fa.std(ddof=1)), fmt(np.ptp(fa)), fmt(fb.mean()), fmt(fb.std(ddof=1)), fmt(np.ptp(fb)), fmt((fb - fa).mean())))
    (out / "tables.md").write_text("\n".join(lines + summary + detail) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
