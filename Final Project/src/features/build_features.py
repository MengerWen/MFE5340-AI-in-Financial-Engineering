"""Feature manifest helpers for the characteristic file folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.loaders import list_feature_files


@dataclass(frozen=True)
class FeatureManifest:
    """A lightweight manifest for one feature universe."""

    universe: str
    files: tuple[Path, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(path.stem for path in self.files)


def build_feature_manifest(universe: str = "features500", root: Path | None = None) -> FeatureManifest:
    """Create a filename-based feature manifest without loading feature values."""

    return FeatureManifest(universe=universe, files=tuple(list_feature_files(universe, root=root)))
