import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main import get_arguments, write_metadata
from process_audit import ProcessAuditRecorder
import test_joint_contract as contract_tests
from tools.analyze_process_accuracy import audit_snapshots, scan_invariants
from visual_trace import TraceRecorder


class JointInterfaceTests(unittest.TestCase):
    def test_joint_exports_can_be_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = get_arguments(["--action_mode", "joint", "--pricing_alg", "TD3_MLP"])
            args.num_zone = 2
            write_metadata(args, temp)
            metadata = json.loads((root / "metadata.json").read_text())
            env = contract_tests.JointContractTests().environment()
            recorder = ProcessAuditRecorder(root, 1, 3, random_sample_count=0)
            with TraceRecorder(root / "trace", metadata) as trace:
                for t in range(3):
                    price, compensation = np.array([.5, 3.]), np.array([2., .5])
                    demand = np.array([[0, 1], [0, 0]]) if t == 0 else np.zeros((2, 2), int)
                    outcome = env.step(demand, [(1, 0)] if t == 1 else [], price, compensation)
                    metrics = dict(zip(["profit", "expense", "served", "left", "occu_mile", "occu_minute", "empty_mile", "empty_minute"], outcome))
                    metrics.update(reward=outcome[0]-outcome[1], act_veh=env.active_veh)
                    recorder.record_step(t, price, metrics, env)
                    trace.record_step(t, price, metrics, env, compensation)
            recorder.close()
            steps = pd.read_csv(root / "step_audit_log_test_1.csv")
            zones = pd.read_csv(root / "zone_step_log_test_1.csv")
            rows = scan_invariants(steps, metadata, 1) + audit_snapshots(root, 1, zones, metadata, set())
            self.assertFalse([r for r in rows if r["status"] == "FAIL"])
            self.assertIn("driver_payment_formula", {r["rule_name"] for r in rows})
            frame = json.loads((root / "trace" / "steps.jsonl").read_text().splitlines()[1])
            self.assertEqual(frame["policy"], [.5, 3.])
            self.assertEqual(frame["compensation_policy"], [2., .5])
            event = json.loads((root / "trace" / "order_events.jsonl").read_text().splitlines()[0])
            self.assertGreater(event["driver_order_payment"], event["fare"])
            # A missing q history must never silently fall back to p in joint audits.
            snapshot = next((root / "snapshots").glob("*.npz"))
            with np.load(snapshot) as payload:
                data = {key: payload[key] for key in payload.files if key != "compensation_history_before"}
            np.savez_compressed(snapshot, **data)
            with self.assertRaisesRegex(ValueError, 'compensation history'):
                audit_snapshots(root, 1, zones, metadata, set())

    def test_existing_ui_loader_keeps_16_zone_price_policy(self):
        ui = Path(__file__).resolve().parents[2] / "RL_pricing_tools" / "visual_ui"
        if not (ui / "data_loader.py").exists():
            self.skipTest("independent UI repository not installed")
        sys.path.insert(0, str(ui))
        try:
            spec = importlib.util.spec_from_file_location("joint_readonly_ui_loader", ui / "data_loader.py")
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "test_log_1.csv"
                pd.DataFrame([{"t": 0, "policy": np.ones(16), "compensation_policy": np.ones(16)*2,
                               "profit": 10., "expense": 12., "reward": -2., "compensation_mean": 2.}]).to_csv(path, index=False)
                result = loader.load_test_log(str(path))
                self.assertEqual(result["num_zone"], 16)
                self.assertEqual(result["grid_size"], 4)
                self.assertEqual(result["rows"][0]["policy"], [1.]*16)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
