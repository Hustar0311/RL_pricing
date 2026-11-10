"""Economic controls shared by the policy, simulator and export contracts."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EconomicAction:
    price: np.ndarray
    compensation: np.ndarray


@dataclass(frozen=True)
class ActionSpec:
    num_zone: int
    mode: str = "price_only"
    compensation_min: float = 0.5
    compensation_max: float = 2.0

    def __post_init__(self):
        if self.mode not in {"price_only", "joint"}:
            raise ValueError("action_mode must be price_only or joint")
        if self.num_zone <= 0:
            raise ValueError("num_zone must be positive")
        if not (np.isfinite(self.compensation_min) and np.isfinite(self.compensation_max)
                and 0 < self.compensation_min < self.compensation_max):
            raise ValueError("compensation bounds must be finite and 0 < min < max")

    @property
    def dimension(self):
        return self.num_zone * (2 if self.mode == "joint" else 1)

    def decode(self, normalized):
        value = np.asarray(normalized, dtype=float).reshape(-1)
        if value.size != self.dimension or not np.all(np.isfinite(value)):
            raise ValueError("invalid normalized action shape or non-finite action")
        value = np.clip(value, -1.0, 1.0)
        price = np.power(3.0, value[:self.num_zone])
        if self.mode == "joint":
            compensation = np.exp(np.log(self.compensation_min)
                + (value[self.num_zone:] + 1.0) * 0.5
                * (np.log(self.compensation_max) - np.log(self.compensation_min)))
        else:
            compensation = price.copy()
        return EconomicAction(price, compensation)

    def metadata(self):
        return {
            "version": 1, "state_version": 1 if self.mode == "joint" else 0,
            "mode": self.mode, "num_zone": self.num_zone,
            "dimension": self.dimension,
            "layout": ["price", "compensation"] if self.mode == "joint" else ["price"],
            "price_bounds": [1.0 / 3.0, 3.0],
            "compensation_bounds": [self.compensation_min, self.compensation_max]
                if self.mode == "joint" else [1.0 / 3.0, 3.0],
            "transform": "logarithmic", "lock_time": "request",
            "region": "passenger_origin",
            "driver_income_signal": "order_compensation_divided_by_delta"
                if self.mode == "joint" else "passenger_revenue",
            "driver_income_unit": "legacy_reservation_wage_proxy",
        }


def validate_control(value, num_zone, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (num_zone,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("{} must be a positive finite vector of length {}".format(name, num_zone))
    return value.copy()
