"""Training protocol definitions using pandas time-index utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainingProtocol:
    """High-level out-of-sample training protocol."""

    scheme: str = "expanding"
    initial_train_months: int = 60
    validation_months: int = 12
    refit_frequency_months: int = 1
    seed: int = 20260408


def validate_training_protocol(protocol: TrainingProtocol) -> None:
    """Validate rolling/expanding protocol settings."""

    if protocol.scheme not in {"expanding", "rolling"}:
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if protocol.initial_train_months <= 0:
        raise ValueError("initial_train_months must be positive")
    if protocol.validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if protocol.refit_frequency_months <= 0:
        raise ValueError("refit_frequency_months must be positive")


def make_oos_schedule(months: Iterable[pd.Timestamp] | pd.Series, protocol: TrainingProtocol) -> pd.DataFrame:
    """Create a pandas DataFrame describing OOS refit/evaluation months."""

    validate_training_protocol(protocol)
    month_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(months)).dropna().unique())).sort_values()
    rows: list[dict[str, pd.Timestamp]] = []
    start = protocol.initial_train_months + protocol.validation_months
    for pos in range(start, len(month_index), protocol.refit_frequency_months):
        train_start_pos = 0 if protocol.scheme == "expanding" else max(0, pos - protocol.initial_train_months - protocol.validation_months)
        rows.append(
            {
                "train_start": month_index[train_start_pos],
                "train_end": month_index[pos - protocol.validation_months - 1],
                "validation_start": month_index[pos - protocol.validation_months],
                "validation_end": month_index[pos - 1],
                "test_month": month_index[pos],
            }
        )
    return pd.DataFrame(rows)


def run_training(*_args: object, protocol: TrainingProtocol | None = None, **_kwargs: object) -> object:
    """Validate training settings; actual model fitting is reserved for later stages."""

    validate_training_protocol(protocol or TrainingProtocol())
    raise NotImplementedError("Actual model training is reserved for the modeling stage.")
