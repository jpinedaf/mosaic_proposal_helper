import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.units import Quantity


@u.quantity_input
def get_offsets_norotation(
    width: Quantity[u.degree],
    height: Quantity[u.degree],
    pb: Quantity[u.degree],
) -> list:
    """
    Calculate the offsets for the pointings based on the width and height.

    Parameters:
    width (float): Width of the field of view in degrees.
    height (float): Height of the field of view in degrees.
    pb (float): Primary beam size in degrees.

    Returns:
    list: A list of tuples containing the offsets (RA_offset, Dec_offset) for each pointing.
    """
    separation = (pb).to_value(u.deg) / 2.0
    if separation <= 0:
        raise ValueError("pb must be > 0")

    half_width = (width / 2).to_value(u.deg)
    half_height = (height / 2).to_value(u.deg)

    y_step = separation * np.sqrt(3) / 2

    if y_step <= 0:
        raise ValueError("Invalid vertical spacing computed from pb")

    def _offsets_for_row_phase(row_offset: float) -> list:
        max_row = int(np.ceil((half_height / y_step) + 0.5))
        offsets = []

        for row in range(-max_row, max_row + 1):
            dec_offset = (row + row_offset) * y_step
            if abs(dec_offset) > half_height + 1e-12:
                continue

            row_shift = 0.0 if row % 2 == 0 else separation / 2
            max_col = int(np.ceil((half_width + separation / 2) / separation))

            for col in range(-max_col, max_col + 1):
                ra_offset = col * separation + row_shift
                if abs(ra_offset) <= half_width + 1e-12:
                    offsets.append((ra_offset * u.deg, dec_offset * u.deg))

        return offsets

    def _vertical_margin(offsets: list) -> float:
        if not offsets:
            return np.inf
        max_abs_dec = max(abs(dec.to_value(u.deg)) for _, dec in offsets)
        return half_height - max_abs_dec

    # odd phase: rows at n*y_step (includes Dec=0)
    offsets_odd = _offsets_for_row_phase(0.0)
    # even phase: rows at (n+0.5)*y_step (Dec=0 between rows)
    offsets_even = _offsets_for_row_phase(0.5)

    margin_odd = _vertical_margin(offsets_odd)
    margin_even = _vertical_margin(offsets_even)

    # Choose the phase that places outer rows closer to the top/bottom limits.
    # In ties, keep the odd phase to preserve the previous behavior.
    if margin_even < margin_odd:
        return offsets_even
    return offsets_odd


@u.quantity_input
def get_offsets(
    width: Quantity[u.degree],
    height: Quantity[u.degree],
    pb: Quantity[u.degree],
    pa: Quantity[u.degree] = 0 * u.degree,
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
        ra_rot = ra_val * cos_pa + dec_val * sin_pa
        dec_rot = -ra_val * sin_pa + dec_val * cos_pa
        rotated_offsets.append((ra_rot * u.deg, dec_rot * u.deg))

    return rotated_offsets


@u.quantity_input
def compute_pointings(
    ra: Quantity[u.degree],
    dec: Quantity[u.degree],
    width: Quantity[u.degree],
    height: Quantity[u.degree],
    pb: Quantity[u.degree],
    pa: Quantity[u.degree] = 0 * u.degree,
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
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    offset = get_offsets(width, height, pb, pa=pa)

    center = SkyCoord(ra=ra_rad, dec=dec_rad, frame="icrs", unit=u.deg)
    pointings = []
    for offset_i in offset:
        coor = center.spherical_offsets_by(offset_i[0], offset_i[1])
        pointings.append(coor)

    return pointings


def export_iram(pointings: list, filename: str = "iram_pointings.txt"):
    """
    Export the pointings to a text file in the format required by IRAM.

    Parameters:
    pointings (list): A list of tuples containing the pointings (dRA, dDec) for the given mosaic.
    filename (str): The name of the output text file. Default is "iram_pointings.txt".
    """
    with open(filename, "w") as f:
        f.write("#dx;dy\n")
        for i, pointing in enumerate(pointings):
            if i < len(pointings) - 1:
                f.write(
                    f"{pointing[0].to_value(u.arcsec):.2f};{pointing[1].to_value(u.arcsec):.2f}\n"
                )
            else:
                f.write(
                    f"{pointing[0].to_value(u.arcsec):.2f};{pointing[1].to_value(u.arcsec):.2f}"
                )


def export_nrao(
    pointings: list, source: str, vlsr: float, filename: str = "nrao_pointings.pst"
):
    """
    Export the pointings to a text file in the format required by NRAO PST system.

    Parameters:
    pointings (list): A list of tuples containing the pointings (absolute RA, Dec) for the given mosaic.
    source (str): The name of the source. The pointing name is constructed as "source"_i where i is the index of the pointing.
    vlsr (float): The LSR velocity in km/s to be included in the output file.
    filename (str): The name of the output text file. Default is "nrao_pointings.pst".
    """
    with open(filename, "w") as f:
        # Catalogue format:
        # sourceName;groupNames;coordSystem;epoch;longitude;latitude;refFrame;convention;velocity;calibrator;
        #
        for i, pointing in enumerate(pointings):
            # if i < len(pointings) - 1:
            ra_c = pointing[0].ra.to_string(sep=":", unit=u.hourangle, precision=3)
            dec_c = pointing[0].dec.to_string(sep=":", precision=3)
            f.write(
                f"{source}_{i}; {source} Pointings; Equatorial; J2000; {ra_c}; {dec_c}; Lsr Kinematic; Radio; {vlsr};  ;\n"
            )
        # else:
        #     f.write(
        #         f"{pointing[0].to_value(u.arcsec):.2f};{pointing[1].to_value(u.arcsec):.2f}"
        #     )
