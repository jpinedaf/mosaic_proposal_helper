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
    __version__ = importlib.metadata.version(
        "mosaic_proposal_helper"
    )  # pragma: no cover
