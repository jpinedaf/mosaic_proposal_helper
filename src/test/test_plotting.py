import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
from conftest import make_sample_image
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.testing.decorators import image_comparison

from mosaic_proposal_helper import pb_interferometer, plot_TdV
from mosaic_proposal_helper.core import compute_pointings
from mosaic_proposal_helper.plotting import plot_circle_wcs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ion()


def test_pb_interferometer():
    # check at the expected frequency
    assert pb_interferometer(219.08899 * u.GHz, telescope="sma") == 50.4 * u.arcsec
    assert pb_interferometer(1.0 * u.GHz, telescope="vla") == 42.0 * u.arcmin
    assert pb_interferometer(72.78382 * u.GHz, telescope="noema") == 64.1 * u.arcsec
    # with proper scaling
    np.testing.assert_approx_equal(
        pb_interferometer(21.908899 * u.GHz, telescope="sma").to(u.arcsec).value,
        504.0,
        significant=4,
    )
    assert (
        pb_interferometer(10.0 * u.GHz, telescope="vla").to(u.arcmin) == 4.2 * u.arcmin
    )
    assert pb_interferometer(7.278382 * u.GHz, telescope="noema") == 641.0 * u.arcsec


def test_pb_interferometer_no_telescope():
    with pytest.raises(ValueError, match="Unsupported telescope"):
        pb_interferometer(72.78382 * u.GHz, telescope="atca")


def _plot_image_helper(
    tmp_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    distance: Quantity[u.pc] = 140 * u.pc,  # type: ignore[reportUnknownMemberType]
    label_col="white",
):
    """Helper function for continuum plotting tests."""
    # dir = tmp_path
    # dir.mkdir(exist_ok=True)
    # file_name = "sample_image.fits"
    # file_link = os.path.join(os.fspath(dir), file_name)
    hdu = make_sample_image()
    rms = 0.1
    seed = 122807528840384100672342137672332424406
    rng = np.random.default_rng(seed)
    data = hdu.data + rng.standard_normal(hdu.data.shape) * rms
    hdu.data = data
    # hdu.writeto(file_link, overwrite=True)

    distance = 100 * u.pc
    wcs = WCS(hdu.header)
    cmap = "Blues"
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=wcs)

    # get pointings
    box_center = SkyCoord(
        "3:00:00.0", "33:00:00.00", frame="icrs", unit=(u.hourangle, u.deg)
    )
    PB = pb_interferometer(
        115 * u.GHz, telescope="noema"
    )  # primary beam size in arcsec
    pa = 56 * u.degree
    box_height = 1.2 * u.arcmin
    box_width = 2.4 * u.arcmin
    radec_points = compute_pointings(
        box_center.ra,
        box_center.dec,
        width=box_width,
        height=box_height,
        pb=PB,
        pa=pa,
    )

    plot_TdV(
        hdu,
        ax=ax,
        cmap=cmap,
        wcs=wcs,
        vmin=vmin,
        vmax=vmax,
        distance=distance,
        label_col="black",
    )
    for p in radec_points:
        plot_circle_wcs(ax, (p.ra, p.dec), radius=PB * 0.5, edgecolor="blue", alpha=0.5)
    fig.tight_layout()


@image_comparison(
    baseline_images=["example_plot_TdV"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
    tol=10,
)
def test_plot_TdV(tmp_path: Path) -> None:
    _plot_image_helper(tmp_path, vmin=-0.3, vmax=1.3)
