import importlib.metadata

from .core import (
    compute_pointings,
    export_iram,
    get_offsets,
    get_offsets_norotation,
)
from .plotting import pb_interferometer, plot_circle, plot_circle_wcs, plot_TdV

__all__ = [
    "get_offsets_norotation",
    "get_offsets",
    "compute_pointings",
    "export_iram",
    "pb_interferometer",
    "plot_circle",
    "plot_circle_wcs",
    "plot_TdV",
]
try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    try:
        __version__ = importlib.metadata.version("mosaic_proposal_helper")
    except importlib.metadata.PackageNotFoundError:
        # Fallback for uninstalled source checkouts without git / generated _version.py
        __version__ = "0.3"
