from astropy import units as u

from mosaic_proposal_helper import compute_pointings, get_offsets
from mosaic_proposal_helper.core import export_nrao


def test_import_and_offsets_non_empty() -> None:
    offsets = get_offsets(
        width=2.4 * u.arcmin,
        height=1.2 * u.arcmin,
        pb=60 * u.arcsec,
        pa=0 * u.deg,
    )
    assert len(offsets) > 0


def test_compute_pointings_matches_offsets_count() -> None:
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


def test_export_nrao(tmp_path) -> None:
    pointings = compute_pointings(
        ra=10.0 * u.deg,
        dec=20.0 * u.deg,
        width=2.4 * u.arcsec,
        height=1.2 * u.arcsec,
        pb=60 * u.arcsec,
        pa=30 * u.deg,
    )
    filename = tmp_path / "nrao_pointings.txt"
    export_nrao(pointings, source="test_source", vlsr=5.0, filename=str(filename))
    assert filename.exists()
    expected_string = "test_source_0; test_source Pointings; Equatorial; J2000; 0:40:00.000; 20:00:00.000; Lsr Kinematic; Radio; 5.0;  ;"
    with open(str(filename), "r", encoding="ascii") as file:
        file_contents = file.read()
    assert file_contents == expected_string + "\n"
