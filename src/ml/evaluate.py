from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    non_zero_mask = np.abs(true) > 1e-9
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / true[non_zero_mask])) * 100)
    else:
        mape = float("nan")
    return {
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)),
        "mape": mape,
    }


def summarise_walk_forward_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "mape": float("nan")}
    frame = pd.DataFrame(records)
    return {column: float(frame[column].mean()) for column in frame.columns}

