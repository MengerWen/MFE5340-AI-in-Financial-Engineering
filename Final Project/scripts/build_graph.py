"""Stage 1 graph-construction entry point.

This validates graph settings only. Graph snapshots require point-in-time panels
and are intentionally deferred to later stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.build_graph import GraphSpec, validate_graph_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate graph construction settings.")
    parser.add_argument("--method", default="return_correlation_knn")
    parser.add_argument("--lookback-months", default=12, type=int)
    parser.add_argument("--k-neighbors", default=10, type=int)
    parser.add_argument("--include-industry-edges", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = GraphSpec(
        method=args.method,
        lookback_months=args.lookback_months,
        k_neighbors=args.k_neighbors,
        include_industry_edges=args.include_industry_edges,
    )
    validate_graph_spec(spec)
    print(f"Graph spec is valid: {spec}")
    print("Graph construction is intentionally deferred to a later stage.")


if __name__ == "__main__":
    main()
