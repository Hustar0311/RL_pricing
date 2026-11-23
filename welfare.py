"""Per-step welfare columns and exact aggregation of them from evaluation logs."""
import numpy as np

WELFARE_COLUMNS = ["consumer_surplus", "driver_total_income", "online_vehicle_minutes",
                   "driver_hourly_wage", "platform_take_rate", "negative_margin_order_count", "driver_idle_cost"]


def step_welfare(env):
    """Welfare fields of the step that was just executed (read from the environment audit)."""
    audit = env.get_last_step_audit()
    return {key: float(audit[key]) for key in WELFARE_COLUMNS}


def aggregate_welfare(frame):
    """Aggregate a test log to totals; rates are ratios of sums, never means of step ratios."""
    revenue = float(frame["profit"].sum())
    income = float(frame["driver_total_income"].sum())
    minutes = float(frame["online_vehicle_minutes"].sum())
    idle = float(frame["driver_idle_cost"].sum()) if "driver_idle_cost" in frame else 0.0
    return {
        "consumer_surplus": float(frame["consumer_surplus"].sum()),
        "driver_total_income": income,
        "driver_idle_cost": idle,
        "driver_hourly_wage": (income - idle) / minutes * 60.0 if minutes > 0 else float("nan"),
        "platform_take_rate": 1.0 - income / revenue if revenue > 0 else float("nan"),
        "negative_margin_order_count": float(frame["negative_margin_order_count"].sum()),
        "online_vehicle_minutes": minutes,
    }
