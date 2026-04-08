"""Stage 1 evaluation entry point.

This lists planned metrics; it does not evaluate model outputs yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import PRIMARY_METRICS


def main() -> None:
    print("Primary evaluation metrics:")
    for metric in PRIMARY_METRICS:
        print(f"- {metric}")


if __name__ == "__main__":
    main()
