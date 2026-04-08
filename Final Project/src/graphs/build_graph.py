"""Graph construction helpers using pandas, scikit-learn, and networkx."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class GraphSpec:
    """Configuration for a stock graph snapshot."""

    method: str = "return_correlation_knn"
    lookback_months: int = 12
    k_neighbors: int = 10
    include_industry_edges: bool = False


def validate_graph_spec(spec: GraphSpec) -> None:
    """Validate graph construction choices before using historical data."""

    if spec.method != "return_correlation_knn":
        raise ValueError("Only return_correlation_knn is implemented at this stage")
    if spec.lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    if spec.k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")


def correlation_knn_edges(returns_window: pd.DataFrame, spec: GraphSpec) -> pd.DataFrame:
    """Build undirected kNN edges from a stock return correlation matrix.

    Parameters
    ----------
    returns_window:
        Wide stock return DataFrame with dates as rows and stock ids as columns.
    spec:
        Graph configuration. Missing returns are filled with each stock's window
        median before computing correlations.
    """

    validate_graph_spec(spec)
    if returns_window.empty:
        raise ValueError("returns_window must be non-empty")

    clean = returns_window.apply(pd.to_numeric, errors="coerce")
    clean = clean.fillna(clean.median()).fillna(0.0)
    corr = clean.corr().fillna(0.0).clip(-1.0, 1.0)
    distance = 1.0 - corr
    np.fill_diagonal(distance.values, 0.0)

    n_neighbors = min(spec.k_neighbors + 1, distance.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")
    nn.fit(distance)
    distances, indices = nn.kneighbors(distance)

    stocks = pd.Index(distance.index.astype(str), name="stock_id")
    records: list[dict[str, object]] = []
    for src_pos, src in enumerate(stocks):
        for dst_pos, dist in zip(indices[src_pos][1:], distances[src_pos][1:]):
            dst = stocks[dst_pos]
            if src == dst:
                continue
            a, b = sorted([src, dst])
            records.append({"source": a, "target": b, "distance": float(dist), "weight": float(1.0 - dist)})
    return pd.DataFrame.from_records(records).drop_duplicates(["source", "target"]).reset_index(drop=True)


def edges_to_networkx(edges: pd.DataFrame) -> nx.Graph:
    """Convert an edge DataFrame to a networkx Graph."""

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight, distance=row.distance)
    return graph


def build_graph_snapshot(*_args: object, spec: GraphSpec | None = None, **_kwargs: object) -> object:
    """Validate settings; full rolling graph snapshots are reserved for graph modeling."""

    validate_graph_spec(spec or GraphSpec())
    raise NotImplementedError("Rolling graph snapshots are reserved for the graph modeling stage.")
