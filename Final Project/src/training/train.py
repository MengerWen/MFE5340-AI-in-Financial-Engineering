"""Training protocol definitions and Stage 1 placeholder runner."""

from __future__ import annotations

from dataclasses import dataclass


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


def run_training(*_args: object, protocol: TrainingProtocol | None = None, **_kwargs: object) -> object:
    """Placeholder for benchmark and graph model training."""

    validate_training_protocol(protocol or TrainingProtocol())
    raise NotImplementedError("Actual model training is outside the Stage 1 scope.")
