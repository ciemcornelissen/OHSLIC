"""Command-line interface for the OHSLIC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from . import data as data_utils
from .models import get_device
from .pipeline import PipelineResult, SegmenterConfig, load_model, run_capture
from .segmentation import SegmentationParams
from .visualization import create_overview_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OHSLIC hyperspectral pipeline")
    parser.add_argument("command", choices=["process"], help="Pipeline action to execute")
    parser.add_argument("--data-dir", default="data", type=Path, help="Directory containing the sample data")
    parser.add_argument(
        "--features-file",
        default=data_utils.DEFAULT_FEATURES_FILE,
        help="Name of the HDF5 file with hyperspectral lines",
    )
    parser.add_argument(
        "--labels-file",
        default=data_utils.DEFAULT_LABELS_FILE,
        help="Name of the HDF5 file with ground-truth labels",
    )
    parser.add_argument(
        "--model-path",
        default="models/pixel_classifier_efficient_0.90_0.79_0.94_0.96.pth",
        type=Path,
        help="Checkpoint to load for inference",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results") / "generated",
        type=Path,
        help="Directory where outputs will be written",
    )
    parser.add_argument("--no-figure", action="store_true", help="Skip Plotly overview generation")
    parser.add_argument("--save-html", action="store_true", help="Persist the Plotly overview as HTML")
    parser.add_argument("--save-png", action="store_true", help="Persist the overview as a PNG snapshot")
    parser.add_argument("--save-npz", action="store_true", help="Persist inference tensors as a compressed NPZ")

    parser.add_argument("--n-clusters", type=int, default=25, help="Initial number of SLIC clusters")
    parser.add_argument("--feature-weight", type=float, default=30.0, help="Weight for spectral similarity")
    parser.add_argument("--spatial-weight", type=float, default=3.0, help="Weight for spatial proximity")
    parser.add_argument("--min-cluster-size", type=int, default=12, help="Minimum cluster size for splitting")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold triggering cluster refinement",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=1_000,
        help="Cluster size threshold triggering refinement",
    )
    parser.add_argument(
        "--downsample-stride",
        type=int,
        default=4,
        help="Stride used to sparsify spectra for SLIC assignments",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size used for local smoothing when applicable",
    )
    parser.add_argument(
        "--disable-confidence",
        action="store_true",
        help="Disable confidence-aware cluster refinement",
    )

    return parser


def run(args: argparse.Namespace) -> PipelineResult:
    if args.command != "process":
        raise ValueError(f"Unsupported command: {args.command}")

    device = get_device()
    lines, _ = data_utils.load_line_stack(args.data_dir, args.features_file, args.labels_file)
    model = load_model(Path(args.model_path), lines.shape[-1], device)

    segmenter_cfg = SegmenterConfig(
        n_clusters=args.n_clusters,
        feature_weight=args.feature_weight,
        spatial_weight=args.spatial_weight,
        min_cluster_size=args.min_cluster_size,
    )
    segmentation_params = SegmentationParams(
        target_clusters=args.n_clusters,
        use_confidence=not args.disable_confidence,
        confidence_threshold=args.confidence_threshold,
        max_cluster_size=args.max_cluster_size,
        window_size=args.window_size,
        downsample_stride=args.downsample_stride,
    )

    result = run_capture(lines, model, device, segmenter_cfg, segmentation_params)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_npz:
        output_path = args.output_dir / "ohslic_output.npz"
        np.savez_compressed(output_path, **result.as_dict())
        print(f"Saved inference tensors to {output_path}")

    if not args.no_figure:
        rgb = data_utils.to_rgb(lines)
        parameter_maps = [result.spectral_parameters[..., i] for i in range(result.spectral_parameters.shape[-1])]
        fig = create_overview_figure(rgb, parameter_maps, result.confidences)
        saved_visual = False
        if args.save_html:
            html_path = args.output_dir / "ohslic_overview.html"
            fig.write_html(str(html_path))
            print(f"Saved overview figure to {html_path}")
            saved_visual = True
        if args.save_png:
            png_path = args.output_dir / "ohslic_overview.png"
            fig.write_image(str(png_path), scale=2)
            print(f"Saved overview snapshot to {png_path}")
            saved_visual = True
        if not saved_visual:
            fig.show()

    timings = {key: round(value, 3) for key, value in sorted(result.timings.items())}
    print("Timings (s):", timings)

    total_time = sum(result.timings.values())
    line_count = result.line_count
    throughput_msgs = []
    if total_time > 0:
        throughput_msgs.append(f"overall={line_count / total_time:.1f}")
    seg_time = result.timings.get("segmentation", 0.0)
    if seg_time > 0:
        throughput_msgs.append(f"segmentation={line_count / seg_time:.1f}")
    inf_time = result.timings.get("inference", 0.0)
    if inf_time > 0:
        throughput_msgs.append(f"inference={line_count / inf_time:.1f}")
    print(f"Lines processed: {line_count} ({', '.join(throughput_msgs)} lines/s)")

    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
