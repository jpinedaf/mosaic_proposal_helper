import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt


@u.quantity_input
def get_offsets_norotation(width: u.degree, height: u.degree, pb: u.degree) -> list:
    """
    Calculate the offsets for the pointings based on the width and height.

    Parameters:
    width (float): Width of the field of view in degrees.
    height (float): Height of the field of view in degrees.
    pb (float): Primary beam size in degrees.

    Returns:
    list: A list of tuples containing the offsets (RA_offset, Dec_offset) for each pointing.
    """
    separation = (pb).to_value(u.deg)
    if separation <= 0:
        raise ValueError("pb must be > 0")

    half_width = (width / 2).to_value(u.deg)
    half_height = (height / 2).to_value(u.deg)

    y_step = separation * np.sqrt(3) / 2

    if y_step <= 0:
        raise ValueError("Invalid vertical spacing computed from pb")

    max_row = int(np.ceil(half_height / y_step))
    offsets = []

    for row in range(-max_row, max_row + 1):
        dec_offset = row * y_step
        if abs(dec_offset) > half_height + 1e-12:
            continue

        row_shift = 0.0 if row % 2 == 0 else separation / 2
        max_col = int(np.ceil((half_width + separation / 2) / separation))

        for col in range(-max_col, max_col + 1):
            ra_offset = col * separation + row_shift
            if abs(ra_offset) <= half_width + 1e-12:
                offsets.append((ra_offset * u.deg, dec_offset * u.deg))

    return offsets


@u.quantity_input
def get_offsets(
    width: u.degree, height: u.degree, pb: u.degree, pa: u.degree = 0 * u.degree
) -> list:
    """
    Calculate the offsets for the pointings based on the width, height, and position angle.

    Parameters:
    width (float): Width of the field of view in degrees.
    height (float): Height of the field of view in degrees.
    pb (float): Primary beam size in degrees.
    pa (float): Position angle in degrees. Default is 0 degrees (no rotation).

    Returns:
    list: A list of tuples containing the offsets (RA_offset, Dec_offset) for each pointing.
    """
    pa_rad = pa.to_value(u.rad)
    offsets_norotation = get_offsets_norotation(width, height, pb)

    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)

    rotated_offsets = []
    for ra_offset, dec_offset in offsets_norotation:
        ra_val = ra_offset.to_value(u.deg)
        dec_val = dec_offset.to_value(u.deg)
        # Ra = Lon, Dec = Lat
        # Rotate the axes
        # c, s = np.cos(pa_angle), np.sin(pa_angle)
        # Ra_pa = -s * Dec + c * Ra
        # Dec_pa = c * Dec + s * Ra
        ra_rot = ra_val * cos_pa + dec_val * sin_pa
        dec_rot = -ra_val * sin_pa + dec_val * cos_pa
        rotated_offsets.append((ra_rot * u.deg, dec_rot * u.deg))

    return rotated_offsets


@u.quantity_input
def compute_pointings(
    ra: u.degree,
    dec: u.degree,
    width: u.degree,
    height: u.degree,
    pa: u.degree,
    pb: u.degree,
) -> list:
    """
    Compute the pointings for a given right ascension (RA), declination (Dec), and field of view (FOV).

    Parameters:
    ra (float): Right ascension in degrees.
    dec (float): Declination in degrees.
    width (float): Width of the field of view in degrees.
    height (float): Height of the field of view in degrees.
    pa (float): Position angle in degrees.
    pb (float): Primary beam size in degrees.

    Returns:
    list: A list of tuples containing the pointings (RA, Dec) for the given FOV.
    """
    # Convert RA and Dec from degrees to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    center = SkyCoord(ra=ra_rad, dec=dec_rad, frame="icrs", unit=u.deg)
    # Calculate the number of pointings needed based on the FOV
    # num_pointings = int(np.ceil(360 / width))

    offset = get_offsets(width, height, pb, pa=pa)

    # frame_center = center.skyoffset_frame(rotation=None)

    pointings = []
    for offset_i in offset:
        coor = center.spherical_offsets_by(offset_i[0], offset_i[1])
        pointings.append(coor)

    # for i in range(num_pointings):
    #     # Calculate the new RA and Dec for each pointing
    #     new_ra = (ra + i * width) % 360
    #     new_dec = dec

    #     pointings.append((new_ra, new_dec))

    return pointings
