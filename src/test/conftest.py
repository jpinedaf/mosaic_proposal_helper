"""Pytest configuration for mosaic_proposal_helper tests."""

from __future__ import annotations

import numpy as np
from astropy.io import fits


def make_sample_image() -> fits.PrimaryHDU:
    im_size = 501
    center = im_size // 2  # 250
    radius = 35
    # 1. Create full 2D coordinate grids centered at (0, 0)
    y, x = np.mgrid[-center : im_size - center, -center : im_size - center]
    radius_map = np.sqrt(x**2 + y**2)
    data = np.ones((im_size, im_size))
    ra0 = 3.0 * 15
    dec0 = 33.0
    data = np.exp(-0.5 * (radius_map / 10.0) ** 2)
    bunit_str = ("mJy/Beam km/s", "Integrated intensity unit")

    header = fits.Header()
    header["CRVAL1"] = ra0
    header["CRVAL2"] = dec0
    header["CRPIX1"] = 251
    header["CRPIX2"] = 251
    header["CDELT1"] = -150.0 / 200 * (1.0 / 3600.0)  # in degrees
    header["CDELT2"] = 150.0 / 200 * (1.0 / 3600.0)
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["EQUINOX"] = 2000.0
    header["RADESYS"] = ("FK5", "Coordinate system")
    header["RESTFREQ"] = (72.78382e9, "Hz")
    header["BUNIT"] = bunit_str
    header["BMAJ"] = 5.0 / 3600.0  # in degrees
    header["BMIN"] = 3.0 / 3600.0  # in degrees
    header["BPA"] = 22.0
    hdu = fits.PrimaryHDU(data=data, header=header)
    return hdu
