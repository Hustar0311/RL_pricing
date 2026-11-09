from pathlib import Path

import numpy as np
import pandas as pd


class ProcessAuditRecorder:
    """Write auditable test-time state summaries without changing simulation flow."""

    def __init__(self, result_dir, test_id, total_steps, random_sample_count=7, sample_seed=2026):
        self.result_dir = Path(result_dir)
        self.test_id = int(test_id)
        self.total_steps = int(total_steps)
        self.step_rows = []
        self.zone_rows = []
        self.snapshots_dir = self.result_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        fixed = {step for step in [0, 360, 720, 1080] if 0 <= step < self.total_steps}
        rng = np.random.default_rng(int(sample_seed))
        count = min(max(int(random_sample_count), 0), self.total_steps)
        random_steps = set(rng.choice(self.total_steps, size=count, replace=False).tolist()) if count else set()
        self.requested_steps = fixed | random_steps
        self._snapshots = {}
        self._extrema = {}

    @staticmethod
    def _copy_snapshot(audit):
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in audit.items()
        }

    def _retain_snapshot(self, t, audit):
        if t in self.requested_steps:
            self._snapshots[t] = self._copy_snapshot(audit)

        criteria = {
            "demand_peak": (float(np.sum(audit["potential_demand_matrix"])), "max"),
            "active_driver_min": (float(audit["active_driver_count"]), "min"),
            "price_mean_max": (float(np.mean(audit["price_vector"])), "max"),
            "price_mean_min": (float(np.mean(audit["price_vector"])), "min"),
            "enter_rate_max": (float(audit["passenger_enter_rate"]), "max"),
            "enter_rate_min": (float(audit["passenger_enter_rate"]), "min"),
        }
        for name, (value, direction) in criteria.items():
            current = self._extrema.get(name)
            better = current is None or (direction == "max" and value > current[0]) or (direction == "min" and value < current[0])
            if better:
                self._extrema[name] = (value, t, self._copy_snapshot(audit))

    def record_step(self, t, policy, metrics, env):
        stats = env.get_last_step_stats()
        audit = env.get_last_step_audit()
        audit.update(stats)
        audit.update(metrics)
        policy = np.asarray(policy, dtype=float)

        waiting_before = float(np.sum(audit["waiting_request_tensor_before"]))
        waiting_after = float(np.sum(audit["waiting_request_tensor_after"]))
        served = float(stats["served_order_count"])
        actual = float(stats["actual_passenger_count"])
        left = float(metrics["left"])
        request_expected_after = waiting_before - served - left + actual
        state_total = (
            audit["idle_vehicle_count"] + audit["busy_vehicle_count"]
            + audit["cruising_vehicle_count"] + audit["offline_vehicle_count"]
        )
        row = {
            "test_id": self.test_id,
            "step": int(t),
            "passenger_entry_br_enabled": bool(audit["passenger_entry_br_enabled"]),
            "driver_online_br_enabled": bool(audit["driver_online_br_enabled"]),
            "reward": float(metrics["reward"]),
            "profit": float(metrics["profit"]),
            "expense": float(metrics["expense"]),
            "price_mean": float(np.mean(policy)),
            "price_min": float(np.min(policy)),
            "price_max": float(np.max(policy)),
            "potential_passenger_count": float(stats["potential_passenger_count"]),
            "actual_passenger_count": actual,
            "passenger_enter_rate": float(stats["passenger_enter_rate"]),
            "passenger_balk_count": float(stats["passenger_balk_count"]),
            "potential_driver_count": float(stats["potential_driver_count"]),
            "active_driver_count": float(stats["active_driver_count"]),
            "driver_online_rate": float(stats["driver_online_rate"]),
            "idle_vehicle_count": int(audit["idle_vehicle_count"]),
            "busy_vehicle_count": int(audit["busy_vehicle_count"]),
            "cruising_vehicle_count": int(audit["cruising_vehicle_count"]),
            "offline_vehicle_count": int(audit["offline_vehicle_count"]),
            "available_idle_vehicle_count_before": int(np.sum(audit["idle_supply_vector"])),
            "matched_order_count": float(stats["matched_order_count"]),
            "served_order_count": served,
            "left_passenger_count": left,
            "waiting_request_count_before": waiting_before,
            "waiting_request_count_after": waiting_after,
            "empty_mile": float(metrics["empty_mile"]),
            "total_vehicle_count_check": int(state_total),
            "total_request_count_check": waiting_after,
            "request_expected_after": request_expected_after,
            "request_balance_error": waiting_after - request_expected_after,
            "driver_decision_count": float(stats["driver_decision_count"]),
            "driver_expected_online_decision_count": float(stats["driver_expected_online_decision_count"]),
            "driver_actual_online_decision_count": float(stats["driver_actual_online_decision_count"]),
            "driver_online_decision_variance": float(stats["driver_online_decision_variance"]),
        }
        self.step_rows.append(row)

        potential_from = np.sum(audit["potential_demand_matrix"], axis=1)
        actual_from = np.sum(audit["actual_demand_matrix"], axis=1)
        matched_from = np.sum(audit["matched_matrix"], axis=1)
        matched_to = np.sum(audit["matched_matrix"], axis=0)
        for zone in range(len(policy)):
            self.zone_rows.append({
                "test_id": self.test_id,
                "step": int(t),
                "zone_id": int(zone),
                "price_multiplier": float(policy[zone]),
                "waiting_by_origin": float(audit["waiting_by_origin"][zone]),
                "idle_supply": float(audit["idle_supply"][zone]),
                "active_vehicle_count": float(audit["active_vehicle_vector"][zone]),
                "potential_demand_from_zone": float(potential_from[zone]),
                "actual_demand_from_zone": float(actual_from[zone]),
                "passenger_balk_from_zone": float(potential_from[zone] - actual_from[zone]),
                "congestion_proxy": float(audit["congestion_proxy"][zone]),
                "passenger_enter_probability": float(audit["passenger_enter_probability"][zone]),
                "expected_actual_passenger_from_zone": float(
                    potential_from[zone] * audit["passenger_enter_probability"][zone]
                ),
                "matched_order_from_zone": float(matched_from[zone]),
                "matched_order_to_zone": float(matched_to[zone]),
                "left_request_from_zone": float(audit["left_request_from_zone"][zone]),
            })
        self._retain_snapshot(int(t), audit)

    def close(self):
        for _, (_, t, snapshot) in self._extrema.items():
            self._snapshots.setdefault(t, snapshot)
        pd.DataFrame(self.step_rows).to_csv(
            self.result_dir / f"step_audit_log_test_{self.test_id}.csv", index=False
        )
        pd.DataFrame(self.zone_rows).to_csv(
            self.result_dir / f"zone_step_log_test_{self.test_id}.csv", index=False
        )
        for t, snapshot in sorted(self._snapshots.items()):
            payload = dict(snapshot)
            payload["test_id"] = np.asarray(self.test_id)
            payload["step"] = np.asarray(t)
            np.savez_compressed(self.snapshots_dir / f"test_{self.test_id}_step_{t}.npz", **payload)
