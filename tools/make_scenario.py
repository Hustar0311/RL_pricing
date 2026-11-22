"""Derive a supply-constrained (or otherwise rescaled) scenario from grid_small_dynamic/4/4.

  python tools/make_scenario.py --name fleet3_demand3 --fleet_multiplier 3 --demand_scale 3

writes td/tt/vp/dd_train/dd_test/permutation*.npy plus scenario.json to
<workspace>/scenarios/<name>/ (git-ignored; never inside a repository tree).
  --fleet_multiplier m   each driver is copied m times; copies after the first get profile jitter
  --profile_scale s      multiplies every driver reservation profile (raises/lowers the outside option)
  --demand_scale d       multiplies dd_train and dd_test
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCENARIOS = WORKSPACE / "scenarios"


def build(source, fleet_multiplier, profile_scale, demand_scale, jitter, seed):
    vp = np.load(source / "vp.npy")
    rng = np.random.default_rng(seed)
    copies = [vp]
    for _ in range(fleet_multiplier - 1):
        copies.append(vp * np.clip(1.0 + jitter * rng.standard_normal(len(vp)), 0.05, None))
    new_vp = np.concatenate(copies) * profile_scale
    arrays = {name: np.load(source / (name + ".npy")) for name in
              ["td", "tt", "dd_train", "dd_test", "permutation", "permutation2"]}
    arrays["dd_train"] = arrays["dd_train"] * demand_scale
    arrays["dd_test"] = arrays["dd_test"] * demand_scale
    arrays["vp"] = new_vp
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default=str(ROOT / "grid_small_dynamic" / "4" / "4"))
    parser.add_argument("--fleet_multiplier", type=int, default=1)
    parser.add_argument("--profile_scale", type=float, default=1.0)
    parser.add_argument("--demand_scale", type=float, default=1.0)
    parser.add_argument("--profile_jitter", type=float, default=0.05, help="relative std of the jitter on copied profiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default=str(SCENARIOS))
    args = parser.parse_args(argv)
    if args.fleet_multiplier < 1 or args.profile_scale <= 0 or args.demand_scale <= 0:
        parser.error("fleet_multiplier >= 1, profile_scale > 0 and demand_scale > 0 are required")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("*\n", encoding="utf-8")  # scenarios are generated data, never tracked
    target = root / args.name
    target.mkdir(exist_ok=True)
    arrays = build(Path(args.source), args.fleet_multiplier, args.profile_scale, args.demand_scale,
                   args.profile_jitter, args.seed)
    for name, value in arrays.items():
        np.save(target / (name + ".npy"), value)
    meta = {k: v for k, v in vars(args).items()}
    meta.update(vehicles=int(len(arrays["vp"])), profile_mean=float(arrays["vp"].mean()),
                dd_test_total=float(arrays["dd_test"].sum()))
    (target / "scenario.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return target


if __name__ == "__main__":
    main()
