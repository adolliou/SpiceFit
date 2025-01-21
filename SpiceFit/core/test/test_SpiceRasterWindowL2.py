import pytest

from ..SpiceRasterWindow import SpiceRasterWindowL2
from astropy.io import fits
import numpy as np
import unittest


@pytest.fixture
def hdu():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras"
        "_20220317T002536_V03_100663832-017.fits"
    )  # noqa: E501
    hdu_list = fits.open(url)
    hdu = hdu_list[0]
    yield hdu
    hdu_list.close()


class TestSpiceRasterWindowL2:

    def test_return_point_pixels(self, hdu):

        s1 = SpiceRasterWindowL2(hdu=hdu)
        xx, yy, ll, tt = s1.return_point_pixels()
        assert np.nanmax(xx) == 2
        assert np.nansum(tt) == 0
        assert xx.shape == s1.data.shape

        coords, lambda_, t = s1.wcs.pixel_to_world(xx, yy, ll, tt)
        lambda_ = lambda_.to("angstrom").value
        assert lambda_.shape == s1.data.shape
        xx, yy, ll = s1.return_point_pixels(type="xyl")
        assert np.nanmax(xx) == 2
        assert np.nanmax(ll) == 49

        data = s1.data[0, :, :, :]
        assert xx.shape == data.shape

        (
            xx,
            yy,
        ) = s1.return_point_pixels(type="xy")
        data = s1.data[0, 0, :, :]
        assert np.nanmax(xx) == 2
        assert xx.shape == data.shape
