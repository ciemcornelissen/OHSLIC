"""High-level orchestration for running the OHSLIC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import data as data_utils
from .models import MultiTaskMLP, get_default_config, load_weights
from .segmentation import IncrementalSLIC, SegmentationParams, segment_line


@dataclass(frozen=True)
class SegmenterConfig:
    n_clusters: int = 25
    feature_weight: float = 30.0
    spatial_weight: float = 3.0
    min_cluster_size: int = 12


@dataclass
class PipelineResult:
    spectral_parameters: np.ndarray
    confidences: np.ndarray
    timings: Dict[str, float]
    line_count: int

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "spectral_parameters": self.spectral_parameters,
            "confidences": self.confidences,
        }


def load_model(model_path: Path, input_size: int, device: torch.device) -> MultiTaskMLP:
    config = get_default_config(input_size=input_size)
    model = MultiTaskMLP(config).to(device)
    load_weights(model, model_path, device)
    model.eval()
    return model


def run_capture(
    cube: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    segmenter_cfg: SegmenterConfig,
    segmentation_params: Optional[SegmentationParams] = None,
) -> PipelineResult:
    if segmentation_params is None:
        segmentation_params = SegmentationParams()

    slic = IncrementalSLIC(
        n_clusters=segmenter_cfg.n_clusters,
        feature_weight=segmenter_cfg.feature_weight,
        spatial_weight=segmenter_cfg.spatial_weight,
        min_cluster_size=segmenter_cfg.min_cluster_size,
    )

    predictions = []
    confidences = []
    total_timings: Dict[str, float] = {"segmentation": 0.0, "inference": 0.0}

    for line in cube:
        prediction, confidence, timings = segment_line(line, slic, model, device, segmentation_params)
        predictions.append(prediction)
        confidences.append(confidence)
        for key, value in timings.items():
            total_timings[key] = total_timings.get(key, 0.0) + value

    spectral_parameters = np.stack(predictions, axis=0)
    confidence_map = np.stack(confidences, axis=0)

    line_count = len(predictions)

    return PipelineResult(spectral_parameters, confidence_map, total_timings, line_count)


def load_sample_capture(data_dir: Path) -> np.ndarray:
    lines, _ = data_utils.load_line_stack(data_dir)
    return lines