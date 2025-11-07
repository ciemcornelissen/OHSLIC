"""Data loading helpers for the OHSLIC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_FEATURES_FILE = "hsi_data_line_example.h5"
DEFAULT_LABELS_FILE = "hsi_label_line_example.h5"
RGB_BAND_INDICES: Tuple[int, int, int] = (43, 24, 12)


@dataclass(frozen=True)
class LineSample:
    """Container for a single hyperspectral line and its per-pixel labels."""

    spectra: np.ndarray  # shape (width, bands)
    targets: np.ndarray  # shape (width, label_dim)


class LineDataset(Dataset):
    """Simple Dataset wrapper around a stack of hyperspectral lines."""

    def __init__(self, lines: np.ndarray, labels: np.ndarray):
        if lines.shape[:2] != labels.shape[:2]:
            raise ValueError(
                "Line and label stacks need matching spatial dimensions, "
                f"got {lines.shape[:2]} vs {labels.shape[:2]}"
            )
        self._lines = lines.astype(np.float32, copy=False)
        self._labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self._lines.shape[0]

    def __getitem__(self, index: int) -> LineSample:
        return LineSample(self._lines[index], self._labels[index])

    @property
    def bands(self) -> int:
        return self._lines.shape[-1]

    @property
    def width(self) -> int:
        return self._lines.shape[1]


def load_line_stack(
    data_dir: Path,
    features_file: str = DEFAULT_FEATURES_FILE,
    labels_file: str = DEFAULT_LABELS_FILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the example hyperspectral line stack and labels from HDF5 files."""

    features_path = (data_dir / features_file).resolve()
    labels_path = (data_dir / labels_file).resolve()

    if not features_path.exists():
        raise FileNotFoundError(f"Missing hyperspectral sample: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing label sample: {labels_path}")

    with h5py.File(features_path, "r") as features_h5:
        spectra = np.array(features_h5["data"], dtype=np.float32)
    with h5py.File(labels_path, "r") as labels_h5:
        labels = np.array(labels_h5["label"], dtype=np.float32)

    return spectra, labels


def build_dataset(
    data_dir: Path,
    features_file: str = DEFAULT_FEATURES_FILE,
    labels_file: str = DEFAULT_LABELS_FILE,
) -> LineDataset:
    """Convenience helper to load the shipped sample into a PyTorch dataset."""

    lines, labels = load_line_stack(data_dir, features_file, labels_file)
    return LineDataset(lines, labels)


def stack_lines(lines: Iterable[np.ndarray]) -> np.ndarray:
    """Stack a sequence of line arrays into a cube (rows, cols, bands)."""

    stacked = np.stack(tuple(lines), axis=0)
    return stacked


def to_rgb(
    cube: np.ndarray,
    band_indices: Sequence[int] = RGB_BAND_INDICES,
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> np.ndarray:
    """Project the hyperspectral cube to an 8-bit RGB image for visualisation."""

    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (rows, cols, bands), got {cube.shape}")

    rgb = cube[..., list(band_indices)].astype(np.float32)
    if clip_range is not None:
        low, high = clip_range
        rgb = np.clip(rgb, low, high)
    rgb = np.nan_to_num(rgb)
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)) * 255.0
    return rgb.astype(np.uint8)


def collate_to_torch(batch: Sequence[LineSample]) -> dict:
    """Default collation logic for batching line samples in a DataLoader."""

    spectra = torch.from_numpy(np.stack([sample.spectra for sample in batch], axis=0))
    targets = torch.from_numpy(np.stack([sample.targets for sample in batch], axis=0))
    return {"spectra": spectra, "targets": targets}
