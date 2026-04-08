"""Graph construction specifications for stock relationship structure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphSpec:
    """Configuration for a stock graph snapshot."""

    method: str = "return_correlation_knn"
    lookback_months: int = 12
    k_neighbors: int = 10
    include_industry_edges: bool = False


def validate_graph_spec(spec: GraphSpec) -> None:
    """Validate graph construction choices before using historical data."""

    if spec.lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    if spec.k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")


def build_graph_snapshot(*_args: object, spec: GraphSpec | None = None, **_kwargs: object) -> object:
    """Placeholder for point-in-time graph construction."""

    validate_graph_spec(spec or GraphSpec())
    raise NotImplementedError(
        "Graph snapshots should be built in a later stage using only data "
        "available before the prediction month."
    )
