import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units import Quantity
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
from matplotlib import patheffects


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    }
)


@u.quantity_input
def pb_noema(freq_obs: Quantity[u.GHz]) -> u.arcsec:
    """
    Primary beam diameter for NOEMA at the observed frequency.
        PB = 64.1 * (72.78382*u.GHz) / freq_obs

    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    return (64.1 * u.arcsec * 72.78382 * u.GHz / freq_obs).decompose()


def plot_circle(ax, center, radius, axis_units=u.deg, **kwargs) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius.to_value(axis_units) * np.cos(theta)
    y = center[1] + radius.to_value(axis_units) * np.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.scatter(center[0], center[1], marker="+", color="red", s=20)
    return


def plot_circle_wcs(
    ax, center, radius, edgecolor="white", lw=1.5, ls=":", **kwargs
) -> None:
    c0 = SphericalCircle(
        (center[0], center[1]),
        radius,
        edgecolor=edgecolor,
        facecolor="none",
        ls=ls,
        transform=ax.get_transform("fk5"),
        lw=lw,
        **kwargs,
    )
    ax.add_patch(c0)
    return


def plot_TdV(
    TdV,
    ax,
    cmap,
    wcs,
    vmin: float | None = None,
    vmax: float | None = None,
    distance: Quantity[u.pc] = 140 * u.pc,
    label_col: str = "white",
) -> None:
    # plot continuum in color
    im = ax.imshow(
        TdV.data,
        origin="lower",
        interpolation="None",
        cmap=cmap,
        alpha=1.0,
        transform=ax.get_transform(wcs),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(110, 620)  # Pixel based zoom
    ax.set_ylim(60, 500)
    length = (5e3 * u.au / distance).to(u.deg, u.dimensionless_angles())
    add_scalebar(ax, length, label=r"5\,000 au", color=label_col, corner="bottom right")
    # add beam
    add_beam(
        ax,
        header=TdV.header,
        frame=False,
        pad=0.4,
        color=label_col,
        corner="bottom left",
    )
    RA = ax.coords[0]
    DEC = ax.coords[1]
    RA.set_axislabel(r"$\alpha$ (J2000)", minpad=0.7)
    DEC.set_major_formatter("dd:mm")
    RA.set_major_formatter("hh:mm:ss")
    DEC.set_axislabel(r"$\delta$ (J2000)", minpad=0.8)
    DEC.set_ticklabel(rotation=90.0, color="black", exclude_overlapping=True)
    RA.set_ticklabel(color="black", exclude_overlapping=True)
    DEC.set_ticks(spacing=120 * u.arcsec, color="black")
    RA.set_ticks(spacing=10.0 * 15 * u.arcsec, color="black")
    RA.display_minor_ticks(True)
    DEC.display_minor_ticks(True)
    DEC.set_minor_frequency(5)
    RA.set_minor_frequency(5)
    return
