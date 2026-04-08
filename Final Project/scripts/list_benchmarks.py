"""List benchmark model specifications for later experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmarks import BENCHMARKS


def main() -> None:
    for spec in BENCHMARKS:
        graph_flag = "graph" if spec.uses_graph else "no graph"
        print(f"{spec.name}: {spec.family}; {graph_flag}; {spec.asset_pricing_role}")


if __name__ == "__main__":
    main()
