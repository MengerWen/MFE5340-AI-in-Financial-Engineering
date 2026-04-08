"""Data inspection utilities for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.data.loaders import get_project_paths


def _describe_obj(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(obj).__name__}
    if hasattr(obj, "shape"):
        out["shape"] = list(obj.shape)
    if hasattr(obj, "index"):
        idx = obj.index
        out["index_type"] = type(idx).__name__
        out["index_names"] = list(idx.names) if hasattr(idx, "names") else None
        out["index_sample"] = [str(x) for x in list(idx[:3])]
        try:
            out["index_min"] = str(idx.min())
            out["index_max"] = str(idx.max())
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["index_minmax_error"] = repr(exc)
    if hasattr(obj, "columns"):
        cols = obj.columns
        out["columns_type"] = type(cols).__name__
        out["columns_count"] = len(cols)
        out["columns_sample"] = [str(x) for x in list(cols[:8])]
        if len(cols) > 0:
            out["dtypes_sample"] = {str(k): str(v) for k, v in obj.dtypes.head(8).items()}
            out["missing_fraction"] = float(obj.isna().mean().mean())
        else:
            out["dtypes_sample"] = {}
            out["missing_fraction"] = None
    return out


def inspect_data(root: Path | None = None) -> dict[str, Any]:
    """Inspect required Stage 2 data files without mutating them."""

    paths = get_project_paths(root)
    data_dir = paths.data_dir
    summary: dict[str, Any] = {}

    for name in [
        "monthly_returns.pkl",
        "mcap.pkl",
        "BLACKLIST.pkl",
        "UNTRADABLE.pkl",
        "csi500_mask_monthly.pkl",
    ]:
        obj = pd.read_pickle(data_dir / name)
        summary[name] = _describe_obj(obj)

    for name in ["risk_free.csv", "FF5.csv", "HXZ.csv"]:
        df = pd.read_csv(data_dir / name)
        summary[name] = _describe_obj(df)
        summary[name]["head_records"] = df.head(3).to_dict(orient="records")

    try:
        with pd.HDFStore(data_dir / "price.h5", mode="r") as store:
            summary["price.h5"] = {"type": "HDF5", "keys": store.keys()}
            for key in store.keys():
                storer = store.get_storer(key)
                summary["price.h5"][key] = {
                    "storer_type": type(storer).__name__,
                    "nrows": getattr(storer, "nrows", None),
                }
    except Exception as exc:
        summary["price.h5"] = {"error": repr(exc)}

    representative = [
        "abnormal_turnover_21.pkl",
        "total_mv.pkl",
        "west_eps_fy1.pkl",
        "return_on_invested_capital_ttm.pkl",
    ]
    for folder in ["features500", "features"]:
        for name in representative:
            path = data_dir / folder / name
            if path.exists():
                summary[f"{folder}/{name}"] = _describe_obj(pd.read_pickle(path))

    mask = pd.read_pickle(data_dir / "csi500_mask_monthly.pkl").astype(bool)
    feature500 = pd.read_pickle(data_dir / "features500" / representative[0])
    feature = pd.read_pickle(data_dir / "features" / representative[0])
    common_idx = feature500.index.intersection(mask.index)
    common_cols = feature500.columns.intersection(mask.columns)
    nonnull = feature500.loc[common_idx, common_cols].notna()
    mask_common = mask.loc[common_idx, common_cols]
    summary["csi500_filter_check"] = {
        "feature500_index_equals_feature_index": bool(feature500.index.equals(feature.index)),
        "features_index_span": [str(feature.index.min()), str(feature.index.max())],
        "features500_index_span": [str(feature500.index.min()), str(feature500.index.max())],
        "common_dates": int(len(common_idx)),
        "common_stocks": int(len(common_cols)),
        "nonnull_inside_mask": int((nonnull & mask_common).sum().sum()),
        "nonnull_outside_mask": int((nonnull & ~mask_common).sum().sum()),
        "mask_true_count": int(mask_common.sum().sum()),
        "feature500_nonnull_count": int(nonnull.sum().sum()),
    }
    return summary


def write_inspection(summary: dict[str, Any], path: Path) -> None:
    """Write inspection results as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)
