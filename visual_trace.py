import json
from datetime import datetime
from pathlib import Path

import numpy as np


TRACE_SCHEMA_VERSION = 1


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class TraceRecorder:
    def __init__(self, trace_dir, metadata=None, trace_interval=1):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_interval = max(int(trace_interval), 1)
        self._files = []

        manifest = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "trace_interval": self.trace_interval,
            "metadata": metadata or {},
            "files": {
                "steps": "steps.jsonl",
                "vehicle_snapshots": "vehicle_snapshots.jsonl",
                "order_events": "order_events.jsonl",
                "vehicle_events": "vehicle_events.jsonl",
            },
        }
        (self.trace_dir / "manifest.json").write_text(
            json.dumps(_jsonable(manifest), indent=2),
            encoding="utf-8",
        )

        self.steps = self._open("steps.jsonl")
        self.vehicle_snapshots = self._open("vehicle_snapshots.jsonl")
        self.order_events = self._open("order_events.jsonl")
        self.vehicle_events = self._open("vehicle_events.jsonl")

    def _open(self, name):
        handle = (self.trace_dir / name).open("w", encoding="utf-8")
        self._files.append(handle)
        return handle

    def _write_jsonl(self, handle, value):
        handle.write(json.dumps(_jsonable(value), separators=(",", ":")) + "\n")

    def record_step(self, t, policy, metrics, env):
        if int(t) % self.trace_interval != 0:
            return

        step_stats = env.get_last_step_stats()
        step_trace = env.get_last_step_trace()
        waiting_by_origin = np.sum(env.pass_count, axis=(0, 2))
        waiting_by_od = np.sum(env.pass_count, axis=0)
        ongoing_veh_count = env.get_ongoing_veh()

        self._write_jsonl(self.steps, {
            "t": int(t),
            "policy": np.asarray(policy, dtype=float),
            "metrics": metrics,
            "bounded_rationality": step_stats,
            "vehicle_count": np.asarray(env.veh_count, dtype=int),
            "waiting_passenger_count": waiting_by_origin.astype(int),
            "waiting_od_count": waiting_by_od.astype(int),
            "ongoing_vehicle_count": ongoing_veh_count.astype(int),
            "active_vehicle_count": int(env.active_veh),
            "avg_profit": float(env.avg_profit),
            "zone_profit": np.asarray(env.zone_profit, dtype=float),
        })

        self._write_jsonl(self.vehicle_snapshots, {
            "t": int(t),
            "vehicles": [
                {
                    "vehicle_id": int(v.id),
                    "zone": int(v.loc),
                    "state": int(v.state),
                    "remaining_time": float(v.remaining_time),
                    "waiting": int(v.waiting),
                    "from_zone": int(getattr(v, "trace_from_zone", v.loc)),
                    "to_zone": int(getattr(v, "trace_to_zone", v.loc)),
                    "total_time": float(getattr(v, "trace_total_time", 0)),
                    "movement_type": getattr(v, "trace_movement_type", "idle"),
                }
                for v in env.veh_list
            ],
        })

        for event in step_trace.get("order_events", []):
            row = dict(event)
            row["t"] = int(t)
            self._write_jsonl(self.order_events, row)

        for event in step_trace.get("vehicle_events", []):
            row = dict(event)
            row["t"] = int(t)
            self._write_jsonl(self.vehicle_events, row)

    def close(self):
        for handle in self._files:
            if not handle.closed:
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
