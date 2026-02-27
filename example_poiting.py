import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
from matplotlib import patheffects

from mosaic import get_offsets, compute_pointings

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
def pb_noema(freq_obs: u.GHz) -> u.arcsec:
    """
    Primary beam diameter for NOEMA at the observed frequency.
        PB = 64.1 * (72.78382*u.GHz) / freq_obs

    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    return (64.1 * u.arcsec * 72.78382 * u.GHz / freq_obs).decompose()


def plot_circle(ax, center, radius, **kwargs) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius.to_value(u.deg) * np.cos(theta)
    y = center[1] + radius.to_value(u.deg) * np.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.scatter(center[0], center[1], marker="+", color="red", s=20)
    return


def plot_circle_wcs(ax, center, radius, **kwargs) -> None:
    # x = center[0] + radius.to_value(u.deg) * np.cos(theta)
    # y = center[1] + radius.to_value(u.deg) * np.sin(theta)
    c0 = SphericalCircle(
        (center[0], center[1]),
        radius,
        edgecolor="white",
        facecolor="none",
        ls=":",
        transform=ax.get_transform("fk5"),
    )
    ax.add_patch(c0)
    # ax.plot(x, y, **kwargs)
    # ax.scatter(center[0], center[1], marker="+", color="red", s=20)
    return


def plot_TdV(
    TdV, ax, cmap, wcs, vmin: float | None = None, vmax: float | None = None
) -> None:
    # plot continuum in color
    im = ax.imshow(
        TdV.data,
        origin="lower",
        interpolation="None",
        cmap=color_map,
        alpha=1.0,
        transform=ax.get_transform(wcs),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(110, 620)  # Pixel based zoom
    ax.set_ylim(60, 500)
    length = (5e3 * u.au / (distance * u.pc)).to(u.deg, u.dimensionless_angles())
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


plt.ion()
# Example usage
PB = pb_noema(115 * u.GHz)
box_height = 3 * u.arcmin
box_width = 5 * u.arcmin
pointings = get_offsets(width=box_width, height=box_height, pb=PB, pa=45 * u.degree)

pointings2 = get_offsets(width=box_width, height=box_height, pb=PB, pa=0 * u.degree)

# example usage without plotting images
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("RA Offset (degrees)")
ax.set_ylabel("Dec Offset (degrees)")
ax.invert_xaxis()

for p in pointings:
    plot_circle(
        ax, (p[0].to_value(u.deg), p[1].to_value(u.deg)), PB, color="blue", alpha=0.5
    )
for p in pointings2:
    plot_circle(
        ax, (p[0].to_value(u.deg), p[1].to_value(u.deg)), PB, color="green", alpha=0.5
    )
ax.set_aspect("equal")


cmap_mom0_default = "inferno"
color_nan = "0.9"
color_map = plt.get_cmap(cmap_mom0_default).copy()
color_map.set_bad(color=color_nan)
vmin = -0.2
vmax = 6.0
TdV_11 = fits.open("B1main_NH3_11_TdV.fits")[0]
wcs_TdV_11 = WCS(TdV_11.header)


box_center = SkyCoord(
    "3:33:00.95", "31:04:20.65", frame="icrs", unit=(u.hourangle, u.deg)
)

radec_points = compute_pointings(
    box_center.ra,
    box_center.dec,
    width=box_width,
    height=box_height,
    pb=PB,
    pa=15 * u.degree,
)


path_effects = [patheffects.withStroke(linewidth=3, foreground="white")]
pos_label = [0.05, 0.95]
label_col = "black"
distance = 300.0  # * u.pc

fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111, projection=wcs_TdV_11)
plot_TdV(TdV_11, ax2, color_map, wcs_TdV_11, vmin=vmin, vmax=vmax)

for p in radec_points:
    plot_circle_wcs(ax2, (p.ra, p.dec), PB, color="blue", alpha=0.5)

ax2.text(
    0.70,
    pos_label[1] - 0.1,
    "B1 main",
    transform=ax2.transAxes,
    fontsize=14,
    path_effects=path_effects,
    verticalalignment="top",
)
ax2.text(
    0.70,
    pos_label[1],
    "NH$_3$ (1,1)",
    transform=ax2.transAxes,
    fontsize=14,
    verticalalignment="top",
    path_effects=path_effects,
)
fig2.tight_layout()
