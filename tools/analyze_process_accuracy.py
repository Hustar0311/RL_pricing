import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOL = 1e-7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"


def parse_args():
    parser = argparse.ArgumentParser(description="Independently audit bounded-rationality process logs and snapshots.")
    parser.add_argument("--result_dir", required=True, help="Leaf test result directory containing audit CSV files.")
    parser.add_argument("--log_type", default="test", choices=["test"])
    parser.add_argument("--test_id", type=int, default=1, help="Test repeat id; use 0 to analyze all available repeats.")
    parser.add_argument("--sample_steps", default="", help="Comma separated subset of saved snapshot steps.")
    parser.add_argument("--random_sample_count", type=int, default=10,
                        help="Accepted for reproducible reporting; snapshot selection occurs during simulation.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for reports and tables (default: modified-worktree outputs/audit).")
    return parser.parse_args()


def load_metadata(result_dir):
    path = result_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": "unknown",
        "min_price_multiplier": 1.0 / 3.0,
        "max_price_multiplier": 3.0,
        "bounded_rationality_parameters": {},
        "simulation_parameters": {},
    }


def mechanism_enabled(metadata, mechanism):
    enabled = metadata.get("enabled_mechanisms")
    if isinstance(enabled, list):
        aliases = {
            "passenger_entry": {"passenger_entry", "passenger_entry_bounded_rationality"},
            "driver_online": {"driver_online", "driver_online_bounded_rationality"},
        }
        return bool(set(enabled) & aliases.get(mechanism, {mechanism}))
    flags = metadata.get("mechanism_flags") or {}
    if mechanism in flags:
        return bool(flags[mechanism])
    return metadata.get("model_family") == "bounded_rationality" or metadata.get("schema_version") in {
        "two_mechanism_bounded_rationality",
        "br_v2_two_mechanisms",
    }


def numeric_parameter(mapping, key, default):
    value = mapping.get(key, default)
    return float(default if value is None else value)


def available_test_ids(result_dir, selected):
    if selected > 0:
        return [selected]
    ids = []
    for path in result_dir.glob("step_audit_log_test_*.csv"):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return sorted(ids)


def result_row(test_id, step, rule, recorded, recomputed, tolerance=TOL, message=""):
    try:
        error = abs(float(recorded) - float(recomputed))
        passed = error <= tolerance
    except (TypeError, ValueError):
        error = np.nan
        passed = False
    return {
        "test_id": int(test_id),
        "step": int(step),
        "rule_name": rule,
        "recorded_value": recorded,
        "recomputed_value": recomputed,
        "absolute_error": error,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "message": message,
    }


def boolean_row(test_id, step, rule, passed, recorded, message):
    return {
        "test_id": int(test_id),
        "step": int(step),
        "rule_name": rule,
        "recorded_value": recorded,
        "recomputed_value": "constraint satisfied" if passed else "constraint violated",
        "absolute_error": 0.0 if passed else np.nan,
        "tolerance": 0.0,
        "status": "PASS" if passed else "FAIL",
        "message": message,
    }


def scan_invariants(step_df, metadata, test_id):
    rows = []
    lower = float(metadata.get("min_price_multiplier", 1.0 / 3.0))
    upper = float(metadata.get("max_price_multiplier", 3.0))
    nonnegative = [
        "potential_passenger_count", "actual_passenger_count", "passenger_balk_count",
        "potential_driver_count", "active_driver_count", "matched_order_count",
        "served_order_count", "left_passenger_count", "empty_mile", "profit", "expense",
        "waiting_request_count_before", "waiting_request_count_after",
    ]
    for _, row in step_df.iterrows():
        step = int(row["step"])
        for field in nonnegative:
            value = float(row[field])
            rows.append(boolean_row(test_id, step, f"{field}_nonnegative", value >= -TOL, value,
                                    f"{field} must be nonnegative."))
        rows.append(boolean_row(test_id, step, "price_lower_bound", row["price_min"] >= lower - TOL,
                                row["price_min"], f"Minimum price multiplier must be >= {lower}."))
        rows.append(boolean_row(test_id, step, "price_upper_bound", row["price_max"] <= upper + TOL,
                                row["price_max"], f"Maximum price multiplier must be <= {upper}."))
        rows.append(boolean_row(test_id, step, "actual_not_above_potential",
                                row["actual_passenger_count"] <= row["potential_passenger_count"] + TOL,
                                row["actual_passenger_count"], "Actual entered demand cannot exceed potential demand."))
        rows.append(result_row(test_id, step, "passenger_balk_balance", row["passenger_balk_count"],
                               row["potential_passenger_count"] - row["actual_passenger_count"]))
        expected_enter_rate = (
            row["actual_passenger_count"] / row["potential_passenger_count"]
            if row["potential_passenger_count"] > 0 else 0.0
        )
        rows.append(result_row(test_id, step, "passenger_enter_rate", row["passenger_enter_rate"],
                               expected_enter_rate, message="Zero potential demand uses rate 0."))
        rows.append(boolean_row(test_id, step, "active_not_above_potential",
                                row["active_driver_count"] <= row["potential_driver_count"] + TOL,
                                row["active_driver_count"], "Active drivers cannot exceed all drivers."))
        expected_online_rate = (
            row["active_driver_count"] / row["potential_driver_count"]
            if row["potential_driver_count"] > 0 else 0.0
        )
        rows.append(result_row(test_id, step, "driver_online_rate", row["driver_online_rate"],
                               expected_online_rate))
        state_total = (row["idle_vehicle_count"] + row["busy_vehicle_count"]
                       + row["cruising_vehicle_count"] + row["offline_vehicle_count"])
        rows.append(result_row(test_id, step, "vehicle_state_conservation",
                               row["potential_driver_count"], state_total,
                               message="idle + busy(service) + cruising(reposition) + offline."))
        rows.append(boolean_row(test_id, step, "match_not_above_waiting_before",
                                row["matched_order_count"] <= row["waiting_request_count_before"] + TOL,
                                row["matched_order_count"],
                                "Matching is performed before current-step entrants join the queue."))
        rows.append(boolean_row(test_id, step, "match_not_above_idle_supply",
                                row["matched_order_count"] <= row["available_idle_vehicle_count_before"] + TOL,
                                row["matched_order_count"], "Only idle vehicles may be scheduled."))
        rows.append(result_row(test_id, step, "served_equals_matched",
                               row["served_order_count"], row["matched_order_count"],
                               message="Scheduled matches are executed as service."))
        rows.append(result_row(test_id, step, "request_queue_balance",
                               row["waiting_request_count_after"], row["request_expected_after"],
                               message="after = before - served - left + actual entrants."))
        rows.append(result_row(test_id, step, "test_reward_definition",
                               row["reward"], row["profit"] - row["expense"],
                               message="Test logs define reward as profit - expense."))
    return rows


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def saved_snapshot_paths(result_dir, test_id, selected_steps):
    paths = sorted((result_dir / "snapshots").glob(f"test_{test_id}_step_*.npz"))
    if not selected_steps:
        return paths
    return [p for p in paths if int(p.stem.rsplit("_", 1)[1]) in selected_steps]


def audit_snapshots(result_dir, test_id, zone_df, metadata, selected_steps):
    rows = []
    br = metadata.get("bounded_rationality_parameters", {})
    sim = metadata.get("simulation_parameters", {})
    metadata_passenger_enabled = mechanism_enabled(metadata, "passenger_entry")
    metadata_driver_enabled = mechanism_enabled(metadata, "driver_online")
    beta_p = numeric_parameter(br, "beta_passenger", 1.0)
    base_u = numeric_parameter(br, "passenger_base_utility", 2.0)
    price_sensitivity = numeric_parameter(br, "passenger_price_sensitivity", 1.0)
    wait_sensitivity = numeric_parameter(br, "passenger_wait_sensitivity", 0.1)
    beta_d = numeric_parameter(br, "beta_driver", 1.0)
    alpha0 = float(sim.get("alpha0", 2.0))
    alpha1 = float(sim.get("alpha1", 2.0))
    alpha2 = float(sim.get("alpha2", 0.1))
    beta1 = float(sim.get("beta1", 0.4))
    beta2 = float(sim.get("beta2", 0.1))
    delta = float(sim.get("delta", 0.75))

    for path in saved_snapshot_paths(result_dir, test_id, selected_steps):
        snapshot = np.load(path)
        step = int(snapshot["step"])
        passenger_enabled = (
            bool(snapshot["passenger_entry_br_enabled"])
            if "passenger_entry_br_enabled" in snapshot.files
            else metadata_passenger_enabled
        )
        driver_enabled = (
            bool(snapshot["driver_online_br_enabled"])
            if "driver_online_br_enabled" in snapshot.files
            else metadata_driver_enabled
        )
        zones = zone_df.loc[zone_df["step"] == step].sort_values("zone_id")
        waiting = snapshot["waiting_request_tensor_before"].sum(axis=(0, 2))
        idle = snapshot["idle_supply_vector"]
        congestion = waiting / (idle + 1.0)
        utility = base_u - price_sensitivity * snapshot["price_vector"] - wait_sensitivity * congestion ** 2
        if passenger_enabled:
            probability = sigmoid(utility / beta_p) if beta_p > TOL else (utility >= 0).astype(float)
            passenger_probability_rule = "passenger_enter_probability_formula"
        else:
            probability = np.ones_like(utility, dtype=float)
            passenger_probability_rule = "passenger_entry_disabled_probability"

        for zone, zone_row in zip(range(len(waiting)), zones.to_dict("records")):
            for rule, recorded, recomputed in [
                ("zone_waiting_by_origin", zone_row["waiting_by_origin"], waiting[zone]),
                ("zone_idle_supply", zone_row["idle_supply"], idle[zone]),
                ("zone_congestion_proxy", zone_row["congestion_proxy"], congestion[zone]),
                (passenger_probability_rule, zone_row["passenger_enter_probability"], probability[zone]),
            ]:
                item = result_row(test_id, step, f"{rule}[zone={zone}]", recorded, recomputed)
                item["snapshot_file"] = str(path)
                rows.append(item)

        potential = snapshot["potential_demand_matrix"]
        actual = snapshot["actual_demand_matrix"]
        rows.append(boolean_row(test_id, step, "od_actual_within_potential",
                                bool(np.all((actual >= 0) & (actual <= potential))), float(np.sum(actual)),
                                "Every actual OD demand entry must be between zero and its potential count."))
        rows.append(result_row(test_id, step, "snapshot_potential_total",
                               float(np.sum(potential)), float(zones["potential_demand_from_zone"].sum())))
        rows.append(result_row(test_id, step, "snapshot_actual_total",
                               float(np.sum(actual)), float(zones["actual_demand_from_zone"].sum())))
        rows.append(result_row(test_id, step, "snapshot_matched_total",
                               float(np.sum(snapshot["matched_matrix"])), float(zones["matched_order_from_zone"].sum())))

        probs = snapshot["driver_decision_probability_vector"]
        mask = np.isfinite(probs)
        if np.any(mask):
            utility_diff = (
                float(snapshot["avg_profit_for_driver_decisions"])
                - snapshot["driver_profile_vector"][mask]
            )
            if driver_enabled:
                expected_prob = sigmoid(utility_diff / beta_d) if beta_d > TOL else (utility_diff >= 0).astype(float)
                driver_probability_rule = "driver_online_probability_formula_max_error"
            else:
                expected_prob = (utility_diff >= 0).astype(float)
                driver_probability_rule = "driver_online_disabled_threshold_max_error"
            rows.append(result_row(test_id, step, driver_probability_rule,
                                   float(np.max(np.abs(probs[mask] - expected_prob))), 0.0))
        else:
            rows.append({
                **boolean_row(test_id, step, "driver_online_probability_formula", True, "no decision", "No driver decision in this step."),
                "status": "INSUFFICIENT_DATA",
            })

        service_origin = snapshot["service_passenger_origin"].astype(int)
        service_dest = snapshot["service_destination"].astype(int)
        vehicle_origin = snapshot["service_vehicle_origin"].astype(int)
        wait_index = snapshot["service_wait_index"].astype(int)
        distance = snapshot["travel_distance_matrix"]
        travel_time = snapshot["travel_time_matrix"]
        history = snapshot["pricing_history_before"]
        joint = metadata.get("action_space", {}).get("mode") == "joint"
        if joint and "compensation_history_before" not in snapshot.files:
            raise ValueError("joint audit snapshot is missing request-locked compensation history")
        compensation_history = snapshot["compensation_history_before"] if "compensation_history_before" in snapshot.files else history
        if len(service_origin):
            base_fares = (
                alpha0 + alpha1 * distance[service_origin, service_dest]
                + beta1 * travel_time[service_origin, service_dest]
            )
            fares = history[wait_index, service_origin] * base_fares
            compensation_basis = compensation_history[wait_index, service_origin] * base_fares
            pickup_costs = alpha2 * distance[vehicle_origin, service_origin] + beta2 * travel_time[vehicle_origin, service_origin]
            service_expense = np.sum(
                pickup_costs + delta * compensation_basis
            )
        else:
            fares = np.asarray([], dtype=float)
            compensation_basis, pickup_costs = np.asarray([], dtype=float), np.asarray([], dtype=float)
            service_expense = 0.0
        if "driver_payment_total" in snapshot.files:
            margins = fares - delta * compensation_basis - pickup_costs
            for rule, recorded, recomputed in [
                ("driver_payment_formula", snapshot["driver_payment_total"], delta * compensation_basis.sum()),
                ("pickup_cost_formula", snapshot["pickup_cost_total"], pickup_costs.sum()),
                ("negative_margin_formula", snapshot["negative_margin_total"], np.maximum(-margins, 0).sum()),
                ("negative_margin_count", snapshot["negative_margin_order_count"], np.sum(margins < 0)),
            ]:
                rows.append(result_row(test_id, step, rule, float(recorded), float(recomputed)))
        if "driver_income_window" in snapshot.files:
            income_basis = compensation_basis.sum() if joint else fares.sum()
            rows.append(result_row(test_id, step, "driver_income_basis_formula", float(snapshot["driver_income_basis"]), float(income_basis)))
            rows.append(result_row(test_id, step, "driver_income_last_observation", float(snapshot["driver_income_window"][-1]),
                                   float(income_basis / snapshot["driver_income_denominator"])))
            rows.append(result_row(test_id, step, "driver_income_rolling_mean", float(snapshot["avg_profit_for_driver_decisions"]),
                                   float(snapshot["driver_income_window"].mean())))
        reposition_origin = snapshot["reposition_origin"].astype(int)
        reposition_dest = snapshot["reposition_destination"].astype(int)
        reposition_expense = np.sum(
            alpha2 * distance[reposition_origin, reposition_dest]
            + beta2 * travel_time[reposition_origin, reposition_dest]
        ) if len(reposition_origin) else 0.0
        cruising_expense = beta2 * float(snapshot["idle_cruising_vehicle_count"])
        recomputed_expense = float(service_expense + reposition_expense + cruising_expense)
        rows.append(result_row(test_id, step, "profit_formula", float(snapshot["profit"]), float(np.sum(fares))))
        rows.append(result_row(test_id, step, "expense_formula", float(snapshot["expense"]), recomputed_expense))
        rows.append(result_row(test_id, step, "reward_formula", float(snapshot["reward"]),
                               float(np.sum(fares)) - recomputed_expense))
    return rows


def statistical_sanity(step_frames, zone_frames, metadata):
    reports = []
    passenger_enabled = mechanism_enabled(metadata, "passenger_entry")
    driver_enabled = mechanism_enabled(metadata, "driver_online")
    for test_id, step_df in step_frames.items():
        zone_df = zone_frames[test_id]
        expected_by_step = zone_df.groupby("step")["expected_actual_passenger_from_zone"].sum()
        actual = step_df.set_index("step")["actual_passenger_count"]
        probability_variance = zone_df.assign(
            variance=zone_df["potential_demand_from_zone"] * zone_df["passenger_enter_probability"]
            * (1.0 - zone_df["passenger_enter_probability"])
        ).groupby("step")["variance"].sum()
        diff = actual - expected_by_step
        total_var = float(probability_variance.sum())
        reports.append({
            "test_id": test_id,
            "process": "passenger_entry_br" if passenger_enabled else "passenger_entry_baseline",
            "sample_count": int(len(diff)),
            "mean_actual": float(actual.mean()),
            "mean_expected": float(expected_by_step.mean()),
            "mean_difference": float(diff.mean()),
            "max_absolute_difference": float(diff.abs().max()),
            "standardized_total_difference": float(diff.sum() / np.sqrt(total_var)) if total_var > TOL else np.nan,
            "status": "PASS" if total_var <= TOL or abs(diff.sum() / np.sqrt(total_var)) <= 4.0 else "WARN",
        })

        expected = step_df["driver_expected_online_decision_count"]
        observed = step_df["driver_actual_online_decision_count"]
        variance = float(step_df["driver_online_decision_variance"].sum())
        driver_diff = observed - expected
        reports.append({
            "test_id": test_id,
            "process": "driver_online_br" if driver_enabled else "driver_online_threshold",
            "sample_count": int(step_df["driver_decision_count"].sum()),
            "mean_actual": float(observed.mean()),
            "mean_expected": float(expected.mean()),
            "mean_difference": float(driver_diff.mean()),
            "max_absolute_difference": float(driver_diff.abs().max()),
            "standardized_total_difference": float(driver_diff.sum() / np.sqrt(variance)) if variance > TOL else np.nan,
            "status": "PASS" if variance <= TOL or abs(driver_diff.sum() / np.sqrt(variance)) <= 4.0 else "WARN",
        })
    return pd.DataFrame(reports)


def write_markdown(output_dir, metadata, invariant_df, sampled_df, statistics_df, failed_df):
    invariant_fails = int((invariant_df["status"] == "FAIL").sum()) if not invariant_df.empty else 0
    sampled_fails = int((sampled_df["status"] == "FAIL").sum()) if not sampled_df.empty else 0
    enabled = metadata.get("enabled_mechanisms") or []
    enabled_text = ", ".join(enabled) if enabled else "none (baseline behavior)"
    invariant_summary = (
        "# Full-Step Invariant Check\n\n"
        f"- Schema: `{metadata.get('schema_version', 'unknown')}`\n"
        f"- Checked rules: {len(invariant_df)}\n"
        f"- Failed checks: {invariant_fails}\n"
        "- Matching order: scheduled requests are drawn from the queue before current-step entrants are enqueued.\n"
        "- Vehicle states: idle + busy(service) + cruising(reposition) + offline.\n"
    )
    (output_dir / "invariant_check_summary.md").write_text(invariant_summary, encoding="utf-8")
    sampled_summary = (
        "# Sampled Snapshot Formula Audit\n\n"
        f"- Checked formulas: {len(sampled_df)}\n"
        f"- Failed checks: {sampled_fails}\n"
        "- Passenger probabilities, driver decision probabilities, and fare/cost formulas are recomputed from NPZ values with NumPy.\n"
    )
    (output_dir / "sampled_step_audit_summary.md").write_text(sampled_summary, encoding="utf-8")
    stats_text = (
        "```text\n" + statistics_df.to_string(index=False) + "\n```"
        if not statistics_df.empty else "No statistical rows."
    )
    report = (
        "# Bounded-Rationality Process Accuracy Audit Report\n\n"
        f"Schema version: `{metadata.get('schema_version', 'unknown')}`\n\n"
        "## Scope\n\n"
        f"Enabled mechanisms: `{enabled_text}`. "
        "Disabled mechanisms are checked against their baseline deterministic behavior. "
        "Scheduled matches are executed as service.\n\n"
        "## Invariants\n\n"
        f"{len(invariant_df)} rule instances checked; {invariant_fails} failed.\n\n"
        "## Sampled Formula Recalculation\n\n"
        f"{len(sampled_df)} formula instances checked; {sampled_fails} failed.\n\n"
        "## Statistical Sanity\n\n"
        f"{stats_text}\n\n"
        "## Failed Cases\n\n"
        f"{len(failed_df)} failed rule instances are listed in `failed_cases.csv`.\n\n"
        "## Interpretation Limit\n\n"
        "Random Bernoulli outcomes are checked statistically over time; individual decisions are not required to equal their probabilities.\n"
    )
    (output_dir / "accuracy_audit_report.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(result_dir)
    selected_steps = {int(x.strip()) for x in args.sample_steps.split(",") if x.strip()}
    test_ids = available_test_ids(result_dir, args.test_id)

    invariant_rows = []
    sampled_rows = []
    step_frames = {}
    zone_frames = {}
    for test_id in test_ids:
        step_path = result_dir / f"step_audit_log_test_{test_id}.csv"
        zone_path = result_dir / f"zone_step_log_test_{test_id}.csv"
        if not step_path.exists() or not zone_path.exists():
            invariant_rows.append(boolean_row(test_id, -1, "audit_input_exists", False, str(step_path),
                                              "Missing step or zone audit input file."))
            continue
        step_df = pd.read_csv(step_path)
        zone_df = pd.read_csv(zone_path)
        step_frames[test_id] = step_df
        zone_frames[test_id] = zone_df
        invariant_rows.extend(scan_invariants(step_df, metadata, test_id))
        sampled_rows.extend(audit_snapshots(result_dir, test_id, zone_df, metadata, selected_steps))

    invariant_df = pd.DataFrame(invariant_rows)
    sampled_df = pd.DataFrame(sampled_rows)
    statistics_df = statistical_sanity(step_frames, zone_frames, metadata) if step_frames else pd.DataFrame()
    failures = []
    if not invariant_df.empty:
        failures.append(invariant_df.loc[invariant_df["status"] == "FAIL"].assign(source="invariant"))
    if not sampled_df.empty:
        failures.append(sampled_df.loc[sampled_df["status"] == "FAIL"].assign(source="snapshot"))
    failed_df = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(
        columns=["test_id", "step", "rule_name", "status", "message", "source"]
    )

    invariant_df.to_csv(output_dir / "invariant_check_report.csv", index=False)
    sampled_df.to_csv(output_dir / "sampled_step_audit_report.csv", index=False)
    statistics_df.to_csv(output_dir / "statistical_sanity_report.csv", index=False)
    failed_df.to_csv(output_dir / "failed_cases.csv", index=False)
    write_markdown(output_dir, metadata, invariant_df, sampled_df, statistics_df, failed_df)
    print(f"Accuracy audit outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
