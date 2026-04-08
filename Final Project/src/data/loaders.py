"""Pandas-based utilities for locating and inspecting project data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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
    """List characteristic files for a universe folder."""

    if universe not in {"features500", "features"}:
        raise ValueError("universe must be 'features500' or 'features'")

    feature_dir = get_project_paths(root).data_dir / universe
    return sorted(feature_dir.glob("*.pkl"))


def summarize_data_tree(root: Path | None = None) -> pd.DataFrame:
    """Return pandas filesystem metadata for the known data layout."""

    paths = get_project_paths(root)
    records: list[dict[str, Any]] = []
    for name, role in EXPECTED_DATA_ROLES.items():
        path = paths.data_dir / name
        files = list(path.glob("*.pkl")) if path.is_dir() else []
        records.append(
            {
                "name": name,
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
                "bytes": path.stat().st_size if path.is_file() else pd.NA,
                "pkl_file_count": len(files) if path.is_dir() else pd.NA,
                "role": role,
            }
        )
    return pd.DataFrame.from_records(records)


def load_pickle_table(path: Path) -> Any:
    """Load a pickle-backed table with pandas."""

    return pd.read_pickle(path)
