"""Reproduce the joint-control CPU matrix and CUDA engineering acceptance.

Run from the algorithm worktree with the training Python environment. All jobs
use this worktree as cwd; retained evidence lives under workspace/artifacts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from tools.analyze_process_accuracy import audit_snapshots, scan_invariants


def runtime_hash():
    value = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def run_command(arguments, log_path):
    env = dict(os.environ, TEMP=str(WORKSPACE/"tmp"), TMP=str(WORKSPACE/"tmp"),
               TMPDIR=str(WORKSPACE/"tmp"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as output:
        result = subprocess.run([sys.executable, "-B"] + arguments, cwd=str(ROOT), env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError("command failed: {} (see {})".format(arguments, log_path))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locate_leaf(scenario, category, name):
    matches = [path.parent for path in (ROOT/"results"/scenario/category).rglob("metadata.json")
               if read_json(path).get("name") == name]
    if len(matches) != 1:
        raise AssertionError("expected one result leaf for {}, got {}".format(name, matches))
    return matches[0]


def policy_vector(value):
    return np.fromstring(str(value).strip("[]").replace(",", " "), sep=" ")


def case_name(case, run_id):
    dataset, algorithm, frequency, device = case
    return "ja{}_{}{}{}{}".format(run_id, dataset[0], "m" if algorithm.endswith("MLP") else "c", frequency,
                                  "g" if device == "cuda" else "c")


def verify_case(case, out, name, source_hash):
    dataset, algorithm, frequency, device = case
    scenario = "grid_small_{}_4_4".format(dataset)
    train_leaf = locate_leaf(scenario, "results", name)
    test_leaf = locate_leaf(scenario, "tests", name)
    metadata = read_json(test_leaf/"metadata.json")
    runtime = read_json(train_leaf/"metadata.json")["runtime"]
    assert runtime["device"] == device, "acceptance result came from the wrong training device"
    if device == "cuda":
        assert runtime["cuda_synchronized"], "CUDA acceptance must execute on CUDA"
    assert metadata["schema_version"] == 2
    assert metadata["action_space"]["dimension"] == 32
    assert metadata["action_space"]["state_version"] == 2
    diagnostics = read_json(train_leaf/"training_diagnostics_2.json")
    assert len(diagnostics) == 2
    for row in diagnostics:
        assert all(np.isfinite(value) for value in row.values())
        assert row["critic_updates"] > 0 and row["actor_updates"] > 0
        assert row["actor_parameter_change"] > 0 and row["critic_parameter_change"] > 0
    training = pd.read_csv(train_leaf/"train_log_2.csv")
    assert len(training) == 2
    np.testing.assert_allclose(training.reward, training.income-training.expense, atol=1e-7)
    expected_steps = len(np.load(ROOT/"grid_small_{}".format(dataset)/"4"/"4"/"dd_test.npy", mmap_mode="r"))
    checks, snapshot_checks, failures = 0, 0, []
    for repeat in range(1, 6):
        frame = pd.read_csv(test_leaf/"test_log_{}.csv".format(repeat))
        assert len(frame) == expected_steps
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        np.testing.assert_allclose(frame.reward, frame.profit-frame.expense, atol=1e-7)
        for field, lower, upper in [("policy", 1/3, 3), ("compensation_policy", .5, 2)]:
            values = np.stack([policy_vector(value) for value in frame[field]])
            assert values.shape == (expected_steps, 16)
            assert np.all(values >= lower-1e-7) and np.all(values <= upper+1e-7)
        step = pd.read_csv(test_leaf/"step_audit_log_test_{}.csv".format(repeat))
        zone = pd.read_csv(test_leaf/"zone_step_log_test_{}.csv".format(repeat))
        assert len(step) == expected_steps and len(zone) == expected_steps*16
        invariants = scan_invariants(step, metadata, repeat)
        snapshots = audit_snapshots(test_leaf, repeat, zone, metadata, set())
        checks += len(invariants)
        snapshot_checks += len(snapshots)
        failures.extend(item for item in invariants+snapshots if item["status"] == "FAIL")
    assert not failures, failures[:8]
    trace_dir = test_leaf/"visual_trace"/"test_log_1"
    trace = [json.loads(line) for line in (trace_dir/"steps.jsonl").read_text().splitlines()]
    assert len(trace) == expected_steps
    assert all(len(row["policy"]) == len(row["compensation_policy"]) == 16 for row in trace)
    ledger = np.zeros((expected_steps, 4))
    for line in (trace_dir/"order_events.jsonl").read_text().splitlines():
        order = json.loads(line)
        fare = order["price_multiplier"] * order["base_fare"]
        payment = .75 * order["compensation_multiplier"] * order["base_fare"]
        margin = fare-payment-order["pickup_cost"]
        np.testing.assert_allclose([order["fare"], order["driver_order_payment"], order["order_margin"]],
                                   [fare, payment, margin], atol=1e-7)
        ledger[order["t"]] += [fare, payment, order["pickup_cost"], max(-margin, 0)]
    first_test = pd.read_csv(test_leaf/"test_log_1.csv")
    np.testing.assert_allclose(ledger, first_test[["profit", "driver_payment_total", "pickup_cost_total", "negative_margin_total"]], atol=1e-7)
    result = {"status": "PASS", "case": case, "name": name, "runtime_source_sha256": source_hash,
              "train_result": str(train_leaf), "test_result": str(test_leaf), "steps_per_epoch": expected_steps,
              "epochs": 2, "test_repeats": 5, "invariant_checks": checks, "snapshot_checks": snapshot_checks,
              "audit_failures": 0, "trace_frames": len(trace), "final_diagnostics": diagnostics[-1]}
    (out/"case.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def execute_case(case, output, run_id, source_hash):
    dataset, algorithm, frequency, device = case
    key = "{}_{}_f{}_{}".format(dataset, algorithm, frequency, device)
    out = output/key
    out.mkdir(exist_ok=True)
    name = case_name(case, run_id)
    base = ["main.py", "-d", "grid_small_{}/4/4/".format(dataset), "--pricing_alg", algorithm,
            "--action_mode", "joint", "--frequency", str(frequency), "--n_epochs", "2",
            "--seed", "47", "--batch_size", "32", "--device", device, "--name", name, "--use_bounded_rationality"]
    print(json.dumps({"case": key, "status": "START"}), flush=True)
    run_command(base+["--mode", "train"], out/"train.log")
    run_command(base+["--mode", "test", "--enable_visual_trace"], out/"evaluate.log")
    result = verify_case(case, out, name, source_hash)
    print(json.dumps({"case": key, "status": "PASS", "critic_updates": result["final_diagnostics"]["critic_updates"],
                      "actor_updates": result["final_diagnostics"]["actor_updates"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument("--workers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--seal_commit", action="store_true", help="Bind completed evidence to an identical committed runtime source")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]{1,24}", args.run_id):
        parser.error("run_id must be 1-24 letters, digits or underscores")
    output = (WORKSPACE/"artifacts"/"joint-price-compensation"/"acceptance"/args.run_id).resolve()
    output.relative_to(WORKSPACE.resolve())
    source_hash = runtime_hash()
    if args.seal_commit:
        manifest = read_json(output/"manifest.json")
        assert manifest["status"] == "PASS" and manifest["runtime_source_sha256"] == source_hash
        manifest["verified_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"verified_commit": manifest["verified_commit"], "runtime_source_sha256": source_hash}))
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["runtime_source_sha256"] != source_hash:
        raise ValueError("runtime source changed; choose a fresh run_id to preserve previous evidence")
    manifest = {"run_id": args.run_id, "status": "RUNNING", "runtime_source_sha256": source_hash,
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "configuration": {"seed": 47, "batch_size": 32, "epochs": 2, "BR": ["passenger_entry", "driver_online"]},
                "results": [], "errors": [], "inputs": {}}
    for dataset in ["static", "dynamic"]:
        for filename in ["dd_train.npy", "dd_test.npy", "td.npy", "tt.npy", "vp.npy"]:
            path = ROOT/"grid_small_{}".format(dataset)/"4"/"4"/filename
            manifest["inputs"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cases = [(data, model, frequency, "cpu") for data in ["static", "dynamic"]
             for model in ["TD3_MLP", "TD3_CNN"] for frequency in [1, 10]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute_case, case, output, args.run_id, source_hash): case for case in cases}
        for future in as_completed(pending):
            try:
                manifest["results"].append(future.result())
            except Exception:
                manifest["errors"].append({"case": pending[future], "error": traceback.format_exc()})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["cuda_available"]:
        try:
            manifest["results"].append(execute_case(("dynamic", "TD3_CNN", 10, "cuda"), output, args.run_id, source_hash))
        except Exception:
            manifest["errors"].append({"case": "CUDA", "error": traceback.format_exc()})
    else:
        manifest["cuda_note"] = "Unavailable on this host"
    manifest["status"] = "FAIL" if manifest["errors"] else "PASS"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "cases_passed": len(manifest["results"]), "manifest": str(manifest_path),
                      "errors": manifest["errors"]}), flush=True)
    if manifest["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
