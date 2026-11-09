import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import bounded_rationality_tag, enabled_mechanisms, get_arguments, write_metadata
from process_audit import ProcessAuditRecorder
from runner import _prepare_demand_batch
from sim import Environment
from tools.analyze_process_accuracy import audit_snapshots, scan_invariants


class MechanismSwitchTests(unittest.TestCase):
    def _environment_config(self, passenger_enabled, driver_enabled):
        return SimpleNamespace(
            use_bounded_rationality=passenger_enabled or driver_enabled,
            use_passenger_entry_br=passenger_enabled,
            use_driver_online_br=driver_enabled,
            br_random_seed=17,
            seed=17,
            beta_passenger=1.0,
            beta_driver=1.0,
            passenger_base_utility=2.0,
            passenger_price_sensitivity=1.0,
            passenger_wait_sensitivity=0.1,
        )

    def test_cli_supports_all_four_mechanism_combinations(self):
        cases = [
            ([], [], ""),
            (["--use_passenger_entry_br"], ["passenger_entry"], "_BR_passenger_entry"),
            (["--use_driver_online_br"], ["driver_online"], "_BR_driver_online"),
            (
                ["--use_bounded_rationality"],
                ["passenger_entry", "driver_online"],
                "_BR_passenger_entry",
            ),
        ]
        for argv, expected_mechanisms, tag_prefix in cases:
            with self.subTest(argv=argv):
                args = get_arguments(argv)
                self.assertEqual(enabled_mechanisms(args), expected_mechanisms)
                self.assertEqual(
                    args.use_bounded_rationality,
                    bool(expected_mechanisms),
                )
                tag = bounded_rationality_tag(args)
                self.assertTrue(tag.startswith(tag_prefix))
                self.assertEqual("passenger_entry" in tag, "passenger_entry" in expected_mechanisms)
                self.assertEqual("driver_online" in tag, "driver_online" in expected_mechanisms)

    def test_passenger_and_driver_paths_are_independent(self):
        travel_distance = np.array([[0.0, 1.0], [1.0, 0.0]])
        travel_time = np.ones((2, 2))
        driver_profiles = np.array([0.0, 2.0, 0.5, 1.5])
        environments = {
            flags: Environment(
                travel_distance,
                travel_time,
                driver_profiles,
                2,
                br_config=self._environment_config(*flags),
            )
            for flags in [(False, False), (True, False), (False, True), (True, True)]
        }

        demand = np.array([[20, 10], [5, 15]])
        price = np.array([1.2, 0.8])
        entered = {
            flags: env.apply_passenger_bounded_rationality(demand, price)
            for flags, env in environments.items()
        }

        np.testing.assert_array_equal(entered[(False, False)], demand)
        np.testing.assert_array_equal(entered[(False, True)], demand)
        np.testing.assert_array_equal(entered[(True, False)], entered[(True, True)])

        baseline_driver_states = [v.state for v in environments[(False, False)].veh_list]
        passenger_only_driver_states = [v.state for v in environments[(True, False)].veh_list]
        driver_only_states = [v.state for v in environments[(False, True)].veh_list]
        both_driver_states = [v.state for v in environments[(True, True)].veh_list]
        self.assertEqual(passenger_only_driver_states, baseline_driver_states)
        self.assertEqual(both_driver_states, driver_only_states)

    def test_driver_only_restores_baseline_price_elasticity(self):
        base_counts = np.array([[[10]]])
        price = np.array([2.0])
        passenger_only = SimpleNamespace(
            use_passenger_entry_br=True,
            use_bounded_rationality=True,
            pricing_mode=False,
        )
        driver_only = SimpleNamespace(
            use_passenger_entry_br=False,
            use_bounded_rationality=True,
            pricing_mode=False,
        )
        self.assertEqual(
            int(_prepare_demand_batch(base_counts, price, passenger_only)[0, 0, 0]),
            10,
        )
        self.assertEqual(
            int(_prepare_demand_batch(base_counts, price, driver_only)[0, 0, 0]),
            5,
        )

    def test_metadata_records_only_active_mechanisms_and_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            args = get_arguments(["--use_driver_online_br"])
            write_metadata(args, str(result_dir))
            metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["enabled_mechanisms"], ["driver_online"])
        self.assertIsNone(metadata["bounded_rationality_parameters"]["beta_passenger"])
        self.assertEqual(metadata["bounded_rationality_parameters"]["beta_driver"], 1.0)
        self.assertEqual(
            metadata["bounded_rationality_parameters"]["rng_streams"],
            {"driver_online": "SeedSequence.spawn(1)"},
        )

    def test_process_audit_accepts_each_mechanism_combination(self):
        travel_distance = np.array([[0.0, 1.0], [1.0, 0.0]])
        travel_time = np.ones((2, 2))
        driver_profiles = np.array([0.0, 2.0, 0.5, 1.5])
        demand = np.array([[3, 1], [1, 2]])
        price = np.array([1.2, 0.8])
        argument_sets = [
            [],
            ["--use_passenger_entry_br"],
            ["--use_driver_online_br"],
            ["--use_bounded_rationality"],
        ]

        for argv in argument_sets:
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as temp_dir:
                result_dir = Path(temp_dir)
                args = get_arguments(argv)
                env = Environment(
                    travel_distance,
                    travel_time,
                    driver_profiles,
                    2,
                    br_config=args,
                )
                actual_demand = env.apply_passenger_bounded_rationality(demand, price)
                outcome = env.step(actual_demand, [], price)
                profit, expense, served, left, occu_mile, occu_minute, empty_mile, empty_minute = outcome
                metrics = {
                    "profit": profit,
                    "expense": expense,
                    "act_veh": env.active_veh,
                    "served": served,
                    "left": left,
                    "occu_mile": occu_mile,
                    "occu_minute": occu_minute,
                    "empty_mile": empty_mile,
                    "empty_minute": empty_minute,
                    "reward": profit - expense,
                }

                write_metadata(args, str(result_dir))
                recorder = ProcessAuditRecorder(
                    result_dir,
                    test_id=1,
                    total_steps=1,
                    random_sample_count=0,
                )
                recorder.record_step(0, price, metrics, env)
                recorder.close()

                metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
                step_df = pd.read_csv(result_dir / "step_audit_log_test_1.csv")
                zone_df = pd.read_csv(result_dir / "zone_step_log_test_1.csv")
                invariant_rows = scan_invariants(step_df, metadata, 1)
                snapshot_rows = audit_snapshots(result_dir, 1, zone_df, metadata, set())
                self.assertFalse([row for row in invariant_rows if row["status"] == "FAIL"])
                self.assertFalse([row for row in snapshot_rows if row["status"] == "FAIL"])


if __name__ == "__main__":
    unittest.main()
