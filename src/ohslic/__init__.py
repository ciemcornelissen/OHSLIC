"""Core package for the OHSLIC hyperspectral clustering pipeline."""

from importlib import metadata

try:
    __version__ = metadata.version("ohslic")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "data",
    "models",
    "pipeline",
    "segmentation",
    "visualization",
]
