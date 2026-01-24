"""Temperature scheduling utility

This module helps adjust temperature throughout an inference run.

The purpose of this is to allow inference to with a higher temperature and gradually reduce it to mitigate precision loss accumulation for longer runs.

"""
from typing import Union, Tuple, Dict, Any
import math

TemperatureSpec = Union[float, Tuple[float, float], Dict[str, Any]]


def parse_temperature_spec(temp_spec: TemperatureSpec) -> Dict[str, Any]:
    """
    Parse temperature specification into normalized config.

    Args:
        temp_spec: Either a float, tuple (start, end), or dict with keys:
            - start: Starting temperature
            - end: Ending temperature
            - schedule: "linear" or "cosine" (default: "linear")

    Returns:
        Dict with keys: start, end, schedule, is_dynamic
    """
    if isinstance(temp_spec, (int, float)):
        return {
            "start": float(temp_spec),
            "end": float(temp_spec),
            "schedule": "linear",
            "is_dynamic": False,
        }
    elif isinstance(temp_spec, tuple) and len(temp_spec) == 2:
        return {
            "start": float(temp_spec[0]),
            "end": float(temp_spec[1]),
            "schedule": "linear",
            "is_dynamic": True,
        }
    elif isinstance(temp_spec, dict):
        start = float(temp_spec.get("start", 1.0))
        end = float(temp_spec.get("end", start))
        return {
            "start": start,
            "end": end,
            "schedule": temp_spec.get("schedule", "linear"),
            "is_dynamic": start != end,
        }
    else:
        raise ValueError(f"Invalid temperature spec: {temp_spec}")


def compute_temperature(
    progress: float,
    start: float,
    end: float,
    schedule: str = "linear",
) -> float:
    """
    Compute temperature at a given progress point.

    Args:
        progress: Generation progress from 0.0 to 1.0
        start: Starting temperature
        end: Ending temperature
        schedule: Interpolation method ("linear" or "cosine")

    Returns:
        Interpolated temperature value
    """
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

    if schedule == "linear":
        return start + (end - start) * progress
    elif schedule == "cosine":
        # Cosine annealing: smooth transition
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'linear' or 'cosine'.")
