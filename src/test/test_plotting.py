from astropy import units as u
import pytest
import numpy as np
from mosaic_proposal_helper import pb_interferometer


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
