"""Incremental hyperspectral superpixel utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from numba import njit


@dataclass(frozen=True)
class SegmentationParams:
    target_clusters: int = 25
    use_confidence: bool = True
    confidence_threshold: float = 0.8
    max_cluster_size: int = 1_000
    window_size: int = 5
    downsample_stride: int = 4


class IncrementalSLIC:
    """Lightweight incremental SLIC variant working line-by-line."""

    def __init__(
        self,
        n_clusters: int,
        feature_weight: float = 1.0,
        spatial_weight: float = 1.0,
        min_cluster_size: int = 3,
    ):
        self.n_clusters = n_clusters
        self.feature_weight = feature_weight
        self.spatial_weight = spatial_weight
        self.min_cluster_size = min_cluster_size
        self.initialized = False

        self.centroids: Optional[np.ndarray] = None
        self.centroid_positions: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.sum_features: Optional[np.ndarray] = None
        self.sum_positions: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.last_features: Optional[np.ndarray] = None
        self.last_positions: Optional[np.ndarray] = None

    def process_line(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected features shape (width, bands); got {features.shape}")

        n_pixels = features.shape[0]
        positions = np.arange(n_pixels, dtype=np.int32)

        if not self.initialized:
            self._initialize_centroids(features, positions)
            self.initialized = True

        labels = _assign_labels_numba(
            features,
            positions,
            self.centroids,  # type: ignore[arg-type]
            self.centroid_positions,  # type: ignore[arg-type]
            self.feature_weight,
            self.spatial_weight,
        )

        self._update_centroids(features, positions, labels)
        self.labels = labels
        self.last_features = features
        self.last_positions = positions
        return labels

    def split_cluster(self, cluster_label: int) -> None:
        if not self.initialized or self.labels is None:
            return
        if cluster_label >= self.n_clusters:
            return

        indices = np.where(self.labels == cluster_label)[0]
        if indices.size < self.min_cluster_size:
            return

        self._remove_cluster_data(cluster_label)
        _adjust_labels_numba(self.labels, cluster_label, -1)

        splits = np.array_split(np.sort(indices), 3)

        for split in splits:
            if split.size == 0:
                continue

            features = self.last_features[split]  # type: ignore[index]
            positions = self.last_positions[split]  # type: ignore[index]

            centroid = features.mean(axis=0).astype(np.float32)
            position = float(positions.mean())
            count = float(split.size)
            sum_features = features.sum(axis=0).astype(np.float32)
            sum_positions = float(positions.sum())

            new_index = len(self.centroids)  # type: ignore[arg-type]
            self.centroids = np.vstack([self.centroids, centroid])  # type: ignore[arg-type]
            self.centroid_positions = np.append(self.centroid_positions, position)  # type: ignore[arg-type]
            self.counts = np.append(self.counts, count)  # type: ignore[arg-type]
            self.sum_features = np.vstack([self.sum_features, sum_features])  # type: ignore[arg-type]
            self.sum_positions = np.append(self.sum_positions, sum_positions)  # type: ignore[arg-type]
            self.labels[split] = new_index  # type: ignore[index]

        self.n_clusters = len(self.centroids)  # type: ignore[arg-type]
        self._order_clusters()

    def remove_cluster(self, cluster_label: Optional[int] = None) -> None:
        if not self.initialized or self.n_clusters <= 1 or self.labels is None:
            return

        if cluster_label is None:
            cluster_label = int(np.argmin(self.counts))  # type: ignore[arg-type]

        indices = np.where(self.labels == cluster_label)[0]
        if indices.size == 0:
            return

        if cluster_label == 0:
            neighbour = 1
        elif cluster_label == self.n_clusters - 1:
            neighbour = self.n_clusters - 2
        else:
            left = abs(self.centroid_positions[cluster_label - 1] - self.centroid_positions[cluster_label])  # type: ignore[index]
            right = abs(self.centroid_positions[cluster_label + 1] - self.centroid_positions[cluster_label])  # type: ignore[index]
            neighbour = cluster_label - 1 if left < right else cluster_label + 1

        self.labels[indices] = neighbour
        self.counts[neighbour] += self.counts[cluster_label]  # type: ignore[index]
        self.sum_features[neighbour] += self.sum_features[cluster_label]  # type: ignore[index]
        self.sum_positions[neighbour] += self.sum_positions[cluster_label]  # type: ignore[index]

        self.centroids[neighbour] = self.sum_features[neighbour] / self.counts[neighbour]  # type: ignore[index]
        self.centroid_positions[neighbour] = self.sum_positions[neighbour] / self.counts[neighbour]  # type: ignore[index]

        self._remove_cluster_data(cluster_label)
        _adjust_labels_numba(self.labels, cluster_label, -1)
        self.n_clusters = len(self.centroids)  # type: ignore[arg-type]
        self._order_clusters()

    def _initialize_centroids(self, features: np.ndarray, positions: np.ndarray) -> None:
        n_pixels = features.shape[0]
        interval = n_pixels / self.n_clusters
        centroid_indices = (interval * np.arange(self.n_clusters) + interval / 2).astype(np.int32)
        centroid_indices = np.clip(centroid_indices, 0, n_pixels - 1)

        self.centroid_positions = positions[centroid_indices]
        self.centroids = features[centroid_indices].astype(np.float32)
        self.counts = np.zeros(self.n_clusters, dtype=np.float32)
        self.sum_features = np.zeros_like(self.centroids)
        self.sum_positions = np.zeros(self.n_clusters, dtype=np.float32)

    def _update_centroids(self, features: np.ndarray, positions: np.ndarray, labels: np.ndarray) -> None:
        n_clusters = self.n_clusters
        n_features = features.shape[1]
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        sum_features = np.zeros((n_clusters, n_features), dtype=np.float32)
        sum_positions = np.zeros(n_clusters, dtype=np.float32)
        _update_sums_numba(features, positions, labels, sum_features, sum_positions, n_clusters)

        self.sum_features += sum_features  # type: ignore[operator]
        self.sum_positions += sum_positions  # type: ignore[operator]
        self.counts += counts  # type: ignore[operator]

        non_zero = self.counts > 0  # type: ignore[operator]
        self.centroids[non_zero] = self.sum_features[non_zero] / self.counts[non_zero][:, None]  # type: ignore[index]
        self.centroid_positions[non_zero] = self.sum_positions[non_zero] / self.counts[non_zero]  # type: ignore[index]

    def _order_clusters(self) -> None:
        order = np.argsort(self.centroid_positions)  # type: ignore[arg-type]
        old_to_new = np.zeros_like(order)
        old_to_new[order] = np.arange(len(order))

        self.centroids = self.centroids[order]
        self.centroid_positions = self.centroid_positions[order]
        self.counts = self.counts[order]
        self.sum_features = self.sum_features[order]
        self.sum_positions = self.sum_positions[order]

        if self.labels is not None:
            self.labels = old_to_new[self.labels]

    def _remove_cluster_data(self, cluster_label: int) -> None:
        mask = np.ones(len(self.centroids), dtype=bool)
        mask[cluster_label] = False
        self.centroids = self.centroids[mask]
        self.centroid_positions = self.centroid_positions[mask]
        self.counts = self.counts[mask]
        self.sum_features = self.sum_features[mask]
        self.sum_positions = self.sum_positions[mask]


def segment_line(
    line: np.ndarray,
    slic: IncrementalSLIC,
    model: torch.nn.Module,
    device: torch.device,
    params: SegmentationParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    start = time.perf_counter()
    labels = slic.process_line(line[:, ::params.downsample_stride])
    timings = {"segmentation": time.perf_counter() - start}

    unique_labels = np.unique(labels)
    label_masks = {label: labels == label for label in unique_labels}
    mean_values = np.stack([line[mask].mean(axis=0) for mask in label_masks.values()]).astype(np.float32)

    inference_start = time.perf_counter()
    with torch.no_grad():
        logits, reg1, reg2, reg3 = model(torch.from_numpy(mean_values).to(device))
        probabilities = torch.softmax(logits, dim=-1)
    timings["inference"] = time.perf_counter() - inference_start

    confidences, classifications = torch.max(probabilities, dim=1)
    regressions = torch.stack([reg1.squeeze(-1), reg2.squeeze(-1), reg3.squeeze(-1)], dim=1)

    output_line = np.zeros((line.shape[0], 3), dtype=np.float32)
    confidence_line = np.zeros(line.shape[0], dtype=np.float32)

    clusters_to_split = set()

    for idx, label in enumerate(unique_labels):
        mask = label_masks[label]
        confidence = float(confidences[idx].cpu())
        confidence_line[mask] = confidence

        if int(classifications[idx].cpu()) == 1:
            prediction = np.zeros(3, dtype=np.float32)
        else:
            prediction = regressions[idx].cpu().numpy().astype(np.float32)

        output_line[mask] = prediction

        if params.use_confidence and confidence < params.confidence_threshold:
            clusters_to_split.add(label)
        if mask.sum() > params.max_cluster_size:
            clusters_to_split.add(label)

    if params.use_confidence:
        for label in sorted(clusters_to_split, reverse=True):
            slic.split_cluster(int(label))
        cluster_diff = slic.n_clusters - params.target_clusters
        if cluster_diff > 0:
            for _ in range(cluster_diff):
                slic.remove_cluster()

    return output_line, confidence_line, timings


@njit
def _assign_labels_numba(features, positions, centroids, centroid_positions, feature_weight, spatial_weight):
    n_pixels = features.shape[0]
    n_clusters = centroids.shape[0]
    labels = np.empty(n_pixels, dtype=np.int32)
    for i in range(n_pixels):
        min_dist = np.inf
        min_label = -1
        for k in range(n_clusters):
            d_feature = 0.0
            for f in range(features.shape[1]):
                diff = features[i, f] - centroids[k, f]
                d_feature += diff * diff
            d_feature = np.sqrt(d_feature)
            d_spatial = abs(positions[i] - centroid_positions[k])
            distance = feature_weight * d_feature + spatial_weight * d_spatial
            if distance < min_dist:
                min_dist = distance
                min_label = k
        labels[i] = min_label
    return labels


@njit
def _update_sums_numba(features, positions, labels, sum_features, sum_positions, n_clusters):
    for i in range(features.shape[0]):
        label = labels[i]
        for j in range(features.shape[1]):
            sum_features[label, j] += features[i, j]
        sum_positions[label] += positions[i]


@njit
def _adjust_labels_numba(labels, threshold, adjustment):
    for i in range(len(labels)):
        if labels[i] > threshold:
            labels[i] += adjustment
