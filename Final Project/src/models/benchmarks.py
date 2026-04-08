"""Benchmark model registry.

These are interfaces only. Actual estimators should be implemented in later
stages after the point-in-time panel is finalized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """High-level benchmark definition for experiment configs."""

    name: str
    family: str
    uses_graph: bool
    asset_pricing_role: str


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("ipca", "linear latent factor", False, "conditional beta benchmark"),
    BenchmarkSpec("conditional_autoencoder", "nonlinear latent factor", False, "nonlinear conditional beta benchmark"),
    BenchmarkSpec("mlp_predictor", "direct prediction", False, "non-graph nonlinear prediction benchmark"),
    BenchmarkSpec("gcn_latent_factor", "graph latent factor", True, "graph-enhanced conditional beta model"),
    BenchmarkSpec("gat_latent_factor", "graph latent factor", True, "attention-based graph conditional beta model"),
)


def benchmark_registry() -> dict[str, BenchmarkSpec]:
    """Return benchmarks keyed by name."""

    return {spec.name: spec for spec in BENCHMARKS}


class BenchmarkModel:
    """Minimal estimator interface shared by later benchmark implementations."""

    def fit(self, *_args: object, **_kwargs: object) -> "BenchmarkModel":
        raise NotImplementedError("Model fitting is intentionally deferred beyond Stage 1.")

    def predict(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("Prediction is intentionally deferred beyond Stage 1.")
