from astropy import units as u

from mosaic_proposal_helper import compute_pointings, get_offsets


def test_import_and_offsets_non_empty():
    offsets = get_offsets(
        width=2.4 * u.arcmin,
        height=1.2 * u.arcmin,
        pb=60 * u.arcsec,
        pa=0 * u.deg,
    )
    assert len(offsets) > 0


def test_compute_pointings_matches_offsets_count():
    pointings = compute_pointings(
        ra=10.0 * u.deg,
        dec=20.0 * u.deg,
        width=2.4 * u.arcmin,
        height=1.2 * u.arcmin,
        pb=60 * u.arcsec,
        pa=30 * u.deg,
    )
    offsets = get_offsets(
        width=2.4 * u.arcmin,
        height=1.2 * u.arcmin,
        pb=60 * u.arcsec,
        pa=30 * u.deg,
    )
    assert len(pointings) == len(offsets)
