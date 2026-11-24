"""Tables from epsilon_decoupling JSON files (v5): Delta at fixed W, online-minutes-floor test, independent repeats.

usage: python tools/report_epsilon.py --input DIR --output DIR/report
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.epsilon_decoupling import boot_mean  # noqa: E402


def days(entry, key):
    return np.array(entry["days"][key], dtype=float)


def cell(b, digits=0):
    return "{:+.{d}f} [{:+.{d}f}, {:+.{d}f}]".format(b["mean"], b["lo"], b["hi"], d=digits)


def ratio_interval(num, den, n_boot, seed=3):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(num), len(num))
        d = den[idx].mean()
        if abs(d) > 1e-9:
            draws.append(num[idx].mean() / d)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def eps_tables(name, doc, n_boot):
    out = ["", "### {}（{}）".format(name, doc["form"]), "",
           "| W | (a) G | (a) 利润 | (b) G | (b) 利润 | Δ=(b)−(a) [95% 区间] | G 在带内 | (a) 在线分钟 | (b) 在线分钟 | (a) 服务量 | (b) 服务量 |", "|" + "---|" * 11]
    for p in doc["points"]:
        d = days(p["b"], "profit") - days(p["a"], "profit")
        ga, gb = days(p["a"], "g_rel").mean(), days(p["b"], "g_rel").mean()
        inband = lambda g: "是" if p["target"] - 0.005 <= g <= p["target"] + doc["band"] + 0.005 else "否"
        out.append("| {:.2f} | {:.3f} | {:.0f} | {:.3f} | {:.0f} | {} | a:{} b:{} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |".format(
            p["target"], ga, days(p["a"], "profit").mean(), gb, days(p["b"], "profit").mean(), cell(boot_mean(d, n_boot)), inband(ga), inband(gb),
            days(p["a"], "online_minutes").mean(), days(p["b"], "online_minutes").mean(), days(p["a"], "served").mean(), days(p["b"], "served").mean()))
    return out


def causal_tables(name, doc, n_boot):
    pts = [p for p in doc["points"] if "b_floor" in p]
    if not pts:
        return []
    out = ["", "#### 因果检验 {}：(b) 加在线分钟下限（≥ 同位置 (a)）".format(name), "",
           "| W | Δ_b=(b)−(a) | Δ_c=(b_c)−(a) | 利润损失 (b)−(b_c) | 可归因于在线分钟差的比例 [95% 区间] | b_c 在线分钟 / (a) | 利润 a / b / b_c | 司机净收入 a / b / b_c | 消费者剩余 a / b / b_c |",
           "|" + "---|" * 9]
    for p in pts:
        pa, pb, pc = (days(p[k], "profit") for k in ("a", "b", "b_floor"))
        loss = pb - pc
        share = loss.mean() / (pb - pa).mean() if abs((pb - pa).mean()) > 1e-9 else float("nan")
        lo, hi = ratio_interval(loss, pb - pa, n_boot)
        f = lambda key: " / ".join("{:.0f}".format(days(p[k], key).mean()) for k in ("a", "b", "b_floor"))
        out.append("| {:.2f} | {} | {} | {:.0f} | {:.0%} [{:.0%}, {:.0%}] | {:.3f} | {} | {} | {} |".format(
            p["target"], cell(boot_mean(pb - pa, n_boot)), cell(boot_mean(pc - pa, n_boot)), loss.mean(), share, lo, hi,
            days(p["b_floor"], "online_minutes").mean() / days(p["a"], "online_minutes").mean(), f("profit"), f("income"), f("consumer_surplus")))
    return out


def repeat_tables(name, doc, n_boot):
    if not doc.get("repeats"):
        return []
    out = ["", "### {}（{}）：5 次独立 ε 约束搜索".format(name, doc["form"]), "",
           "| W | 各次 Δ | 最小 | 中位数 | 最大 | 中位数那次的 bootstrap 区间 | 5 次平均的区间 | 各次 (a) G / (b) G |", "|" + "---|" * 8]
    for r in doc["repeats"]:
        per = [days(x["b"], "profit") - days(x["a"], "profit") for x in r["reps"]]
        means = np.array([d.mean() for d in per])
        med = int(np.argsort(means)[len(means) // 2])
        pooled = np.mean(per, axis=0)
        gs = "; ".join("{:.2f}/{:.2f}".format(days(x["a"], "g_rel").mean(), days(x["b"], "g_rel").mean()) for x in r["reps"])
        out.append("| {:.2f} | {} | {:+.0f} | {:+.0f} | {:+.0f} | {} | {} | {} |".format(
            r["target"], ", ".join("{:+.0f}".format(m) for m in means), means.min(), np.median(means), means.max(),
            cell(boot_mean(per[med], n_boot)), cell(boot_mean(pooled, n_boot)), gs))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# ε 约束结果（v5）", "", "(b) 包含 (a)，评估用新日，Δ 是真实增益的下界；负值读作“检测不到增益”。区间只反映评估日抽样。"]
    for f in sorted(src.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if f.stem.startswith("rep_"):
            lines += repeat_tables(f.stem, doc, args.bootstrap)
        else:
            lines += eps_tables(f.stem, doc, args.bootstrap) + causal_tables(f.stem, doc, args.bootstrap)
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
