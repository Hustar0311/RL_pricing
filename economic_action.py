"""Economic controls shared by the policy, simulator and export contracts."""
from dataclasses import dataclass

import numpy as np

PRICE_ONLY = "price_only"
JOINT = "joint"
FIXED_SHARE = "fixed_share"
LEARNED_SHARE = "learned_share"
JOINT_SHARE = "joint_share"
SHARE_MODES = (FIXED_SHARE, LEARNED_SHARE, JOINT_SHARE)
ACTION_MODES = (PRICE_ONLY, JOINT) + SHARE_MODES
# Every mode except price_only runs through the joint_training runner.
JOINT_RUNNER_MODES = (JOINT,) + SHARE_MODES
DEFAULT_DELTA = 0.75


@dataclass(frozen=True)
class EconomicAction:
    price: np.ndarray
    compensation: np.ndarray
    share: np.ndarray = None


@dataclass(frozen=True)
class ActionSpec:
    num_zone: int
    mode: str = "price_only"
    compensation_min: float = 0.5
    compensation_max: float = 2.0
    share_min: float = 0.2
    share_max: float = 0.95
    share_value: float = 0.75
    delta: float = DEFAULT_DELTA

    def __post_init__(self):
        if self.mode not in ACTION_MODES:
            raise ValueError("action_mode must be one of " + ", ".join(ACTION_MODES))
        if self.num_zone <= 0:
            raise ValueError("num_zone must be positive")
        if not (np.isfinite(self.compensation_min) and np.isfinite(self.compensation_max)
                and 0 < self.compensation_min < self.compensation_max):
            raise ValueError("compensation bounds must be finite and 0 < min < max")
        if self.mode in SHARE_MODES:
            if not (np.isfinite(self.share_min) and np.isfinite(self.share_max)
                    and 0 < self.share_min < self.share_max):
                raise ValueError("share bounds must be finite and 0 < share_min < share_max")
            if not (np.isfinite(self.share_value) and self.share_value > 0):
                raise ValueError("share_value must be finite and positive")
            if not (np.isfinite(self.delta) and self.delta > 0):
                raise ValueError("delta must be finite and positive")

    @classmethod
    def from_args(cls, args, num_zone=None):
        return cls(num_zone if num_zone is not None else getattr(args, "num_zone", 1),
                   getattr(args, "action_mode", PRICE_ONLY),
                   getattr(args, "compensation_min", 0.5), getattr(args, "compensation_max", 2.0),
                   getattr(args, "share_min", 0.2), getattr(args, "share_max", 0.95),
                   getattr(args, "share_value", 0.75))

    @property
    def uses_joint_runner(self):
        return self.mode in JOINT_RUNNER_MODES

    @property
    def is_share_mode(self):
        return self.mode in SHARE_MODES

    @property
    def dimension(self):
        return action_dimension(self.mode, self.num_zone)

    @property
    def state_free_dim(self):
        """Number of trailing action entries that do not depend on the state."""
        return 1 if self.mode == LEARNED_SHARE else 0

    def q_bounds(self):
        """Bounds used to normalise the order-compensation history in the state."""
        if self.mode == JOINT:
            return self.compensation_min, self.compensation_max
        if self.mode == FIXED_SHARE:
            return self.share_value / (3.0 * self.delta), 3.0 * self.share_value / self.delta
        if self.mode in (LEARNED_SHARE, JOINT_SHARE):
            return self.share_min / (3.0 * self.delta), 3.0 * self.share_max / self.delta
        return 1.0 / 3.0, 3.0

    def decode_share(self, value):
        """Map normalized actions in [-1, 1] to a per-zone share vector."""
        Z = self.num_zone
        if self.mode == FIXED_SHARE:
            return np.full(Z, float(self.share_value))
        span = self.share_max - self.share_min
        if self.mode == LEARNED_SHARE:
            return np.full(Z, self.share_min + (value[Z] + 1.0) * 0.5 * span)
        return self.share_min + (value[Z:2 * Z] + 1.0) * 0.5 * span

    def decode(self, normalized):
        value = np.asarray(normalized, dtype=float).reshape(-1)
        if value.size != self.dimension or not np.all(np.isfinite(value)):
            raise ValueError("invalid normalized action shape or non-finite action")
        value = np.clip(value, -1.0, 1.0)
        price = np.power(3.0, value[:self.num_zone])
        share = None
        if self.mode == JOINT:
            compensation = np.exp(np.log(self.compensation_min)
                + (value[self.num_zone:] + 1.0) * 0.5
                * (np.log(self.compensation_max) - np.log(self.compensation_min)))
        elif self.is_share_mode:
            share = self.decode_share(value)
            # q = s * p / delta; written as p * (s / delta) so that s == delta gives q == p exactly.
            compensation = price * (share / self.delta)
        else:
            compensation = price.copy()
        return EconomicAction(price, compensation, share)

    def metadata(self):
        if self.is_share_mode:
            q_low, q_high = self.q_bounds()
            layout = {FIXED_SHARE: ["price"], LEARNED_SHARE: ["price", "share"],
                      JOINT_SHARE: ["price", "share"]}[self.mode]
            return {
                "version": 1, "state_version": 2, "mode": self.mode, "num_zone": self.num_zone,
                "dimension": self.dimension, "layout": layout,
                "price_bounds": [1.0 / 3.0, 3.0],
                "share_bounds": [self.share_min, self.share_max],
                "share_value": self.share_value if self.mode == FIXED_SHARE else None,
                "share_scope": {FIXED_SHARE: "global_fixed", LEARNED_SHARE: "global_state_independent",
                                JOINT_SHARE: "per_zone"}[self.mode],
                "delta": self.delta,
                "compensation_bounds": [q_low, q_high],
                "transform": "logarithmic_price_linear_share", "lock_time": "request",
                "region": "passenger_origin",
                "driver_income_signal": "share_times_price_divided_by_delta",
                "driver_income_unit": "legacy_reservation_wage_proxy",
            }
        return {
            "version": 1, "state_version": 2 if self.mode == "joint" else 0,
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


def action_dimension(mode, num_zone):
    if mode in (JOINT, JOINT_SHARE):
        return 2 * num_zone
    if mode == LEARNED_SHARE:
        return num_zone + 1
    return num_zone


def validate_control(value, num_zone, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (num_zone,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("{} must be a positive finite vector of length {}".format(name, num_zone))
    return value.copy()
