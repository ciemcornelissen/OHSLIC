"""Plotly-based visualisations for OHSLIC outputs."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_overview_figure(
    rgb_image: np.ndarray,
    parameter_maps: Sequence[np.ndarray],
    confidence: np.ndarray | None = None,
    parameter_names: Sequence[str] = ("Chlorophyll", "Carotenoids", "Anthocyanin"),
) -> go.Figure:
    rows = 1 + len(parameter_maps)
    if confidence is not None:
        rows += 1

    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.02)
    fig.add_trace(go.Image(z=_to_uint8(rgb_image)), row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    for idx, (name, array) in enumerate(zip(parameter_names, parameter_maps), start=2):
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(array),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=name),
            ),
            row=idx if confidence is None else idx,
            col=1,
        )
        fig.update_xaxes(visible=False, row=idx, col=1)
        fig.update_yaxes(visible=False, row=idx, col=1)

    if confidence is not None:
        row_idx = rows
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(confidence),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Confidence"),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_xaxes(visible=False, row=row_idx, col=1)
        fig.update_yaxes(visible=False, row=row_idx, col=1)

    fig.update_layout(
        width=1000,
        height=350 * rows,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title="OHSLIC Inference Overview",
    )
    return fig


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    imin, imax = float(np.min(image)), float(np.max(image))
    if imax - imin < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - imin) / (imax - imin)
    return (scaled * 255).clip(0, 255).astype(np.uint8)
