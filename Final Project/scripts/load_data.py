"""Stage 1 data-loading entry point.

Default behavior prints lightweight filesystem metadata. Use --load-pickle in
Stage 2 only after installing requirements and choosing a file to inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_pickle_table, summarize_data_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect project data locations.")
    parser.add_argument("--load-pickle", type=Path, help="Optional pickle path to load with pandas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_pickle is None:
        print(json.dumps(summarize_data_tree(PROJECT_ROOT), indent=2, ensure_ascii=False))
        return

    obj = load_pickle_table(args.load_pickle)
    print(type(obj).__name__)
    if hasattr(obj, "shape"):
        print(f"shape={obj.shape}")


if __name__ == "__main__":
    main()
