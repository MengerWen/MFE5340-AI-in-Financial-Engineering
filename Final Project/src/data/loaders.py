"""Utilities for locating and lightly inspecting project data.

Stage 1 deliberately avoids assuming detailed pickle schemas. Later stages
should load the raw objects after the environment is installed and document
their index/column structure before building panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_DATA_ROLES: dict[str, str] = {
    "monthly_returns.pkl": "monthly stock return matrix; candidate base target source",
    "risk_free.csv": "risk-free rate series for excess-return construction",
    "mcap.pkl": "monthly market capitalization panel",
    "price.h5": "daily adjusted close prices for price-derived features or checks",
    "BLACKLIST.pkl": "date-asset exclusion list",
    "UNTRADABLE.pkl": "date-asset trading-feasibility exclusion list",
    "csi500_mask_monthly.pkl": "monthly CSI 500 membership mask",
    "FF5.csv": "Fama-French five-factor benchmark returns",
    "HXZ.csv": "Hou-Xue-Zhang factor benchmark returns",
    "features500": "CSI 500-filtered monthly characteristic files; main specification",
    "features": "broader monthly characteristic files; robustness specification",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths anchored at the Final Project directory."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


def get_project_root() -> Path:
    """Return the repository subdirectory that contains this project."""

    return Path(__file__).resolve().parents[2]


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """Build project paths without hard-coding absolute locations."""

    return ProjectPaths(root=(root or get_project_root()).resolve())


def list_feature_files(universe: str = "features500", root: Path | None = None) -> list[Path]:
    """List characteristic files for a universe folder.

    Parameters
    ----------
    universe:
        Either ``features500`` for the main CSI 500 spec or ``features`` for
        the broader robustness spec.
    root:
        Optional project root override.
    """

    if universe not in {"features500", "features"}:
        raise ValueError("universe must be 'features500' or 'features'")

    feature_dir = get_project_paths(root).data_dir / universe
    return sorted(feature_dir.glob("*.pkl"))


def summarize_data_tree(root: Path | None = None) -> dict[str, Any]:
    """Return lightweight filesystem metadata for the known data layout."""

    paths = get_project_paths(root)
    data_dir = paths.data_dir
    summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "known_files": {},
        "feature_counts": {},
    }

    for name in EXPECTED_DATA_ROLES:
        path = data_dir / name
        if path.is_file():
            summary["known_files"][name] = {"exists": True, "bytes": path.stat().st_size}
        elif path.is_dir():
            summary["known_files"][name] = {"exists": True, "type": "directory"}
        else:
            summary["known_files"][name] = {"exists": False}

    for universe in ("features500", "features"):
        files = list_feature_files(universe=universe, root=paths.root)
        summary["feature_counts"][universe] = len(files)

    return summary


def load_pickle_table(path: Path) -> Any:
    """Load a pickle-backed table with pandas.

    This helper is intentionally small and dependency-light at import time.
    Stage 2 should call it while documenting the loaded object's exact schema.
    """

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required to load project pickle files. "
            "Install dependencies from requirements.txt first."
        ) from exc

    return pd.read_pickle(path)
