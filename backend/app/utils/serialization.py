from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def to_builtin(value):
    """Convert numpy/pandas values into JSON-safe native Python objects."""
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def round_float(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def safe_list(sequence: Sequence | None) -> list:
    return list(sequence) if sequence is not None else []
