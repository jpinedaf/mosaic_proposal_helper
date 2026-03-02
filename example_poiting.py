import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units import Quantity
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib import patheffects

from mosaic import get_offsets, compute_pointings, export_iram
from mosaic_plotting import plot_TdV, plot_circle, plot_circle_wcs, pb_noema

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

path_effects = [patheffects.withStroke(linewidth=3, foreground="white")]

plt.ion()
plt.close("all")
# Example usage
PB = pb_noema(115 * u.GHz)
pa = 56 * u.degree
box_height = 3.0 * u.arcmin
# box_height = 2.2 * u.arcmin
box_width = 5 * u.arcmin
pointings = get_offsets(width=box_width, height=box_height, pb=PB, pa=pa)

pointings2 = get_offsets(width=box_width, height=box_height, pb=PB, pa=0 * u.degree)

export_iram(pointings, filename="iram_pointings.txt")

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
    pa=pa,
)


pos_label = [0.05, 0.95]
label_col = "black"
distance = 300.0 * u.pc

fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111, projection=wcs_TdV_11)
plot_TdV(
    TdV_11,
    ax2,
    color_map,
    wcs_TdV_11,
    vmin=vmin,
    vmax=vmax,
    distance=distance,
    label_col=label_col,
)

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
