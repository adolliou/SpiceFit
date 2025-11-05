import pytest

from ..SpiceRasterWindow import SpiceRasterWindowL2
from astropy.io import fits
import numpy as np
import unittest
from ..FitResults import FitResults
from ..FittingModel import FittingModel
import os
from pathlib import Path
import astropy.units as u


@pytest.fixture
def hdu1():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras"
        "_20220317T002536_V03_100663832-017.fits"
    )  # noqa: E501
    hdu_list = fits.open(url)
    hdu = hdu_list[0]
    yield hdu
    hdu_list.close()


@pytest.fixture
def hdu2():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras"
        "_20220317T002536_V03_100663832-017.fits"
    )  # noqa: E501
    hdu_list = fits.open(url)
    hdu = hdu_list["Ne VIII 770 - Peak"]
    yield hdu
    hdu_list.close()


@pytest.fixture
def fittemplate():
    return FittingModel(filename="ne_8_770_42_1c.template")


class TestSpiceRasterWindowL2:

    def test_return_point_pixels(self, hdu1):
        s1 = SpiceRasterWindowL2(hdu=hdu1)
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

    def test_average_spectra_over_region(self, hdu2, fittemplate):
        s1 = SpiceRasterWindowL2(hdu=hdu2)

        x = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        y = [450, 450, 450, 451, 451, 451,  452, 452, 452, 453, 453, 453,]  # Here we average over a region of twelve pixels. y is the coordinate along the slit. 
        pixels = (x, y)

        # x, y = np.meshgrid(np.arange(s1.data.shape[3]), np.arange(s1.data.shape[2]))
        # coords = s1.w_xy.pixel_to_world(x, y)
        # x, y = s1.w_xy.world_to_pixel(coords)
        s2 = s1.average_spectra_over_region(pixels=pixels)
        pixels_lims = ((0, 2), (450, 453))
        s6 = s1.average_spectra_over_region(pixels_lims=pixels_lims)
        s = np.nansum(np.abs(s2.data - s6.data)).value

        assert (s < 1.0e-7)
        coords = s1.w_xy.pixel_to_world(x, y)
        s3 = s1.average_spectra_over_region(coords=coords)

        s = np.nansum(np.abs(s3.data - s2.data)).value
        assert (s < 1.0E-7)
        results = FitResults()
        results.fit_spice_window_standard(spicewindow=s2, parallelism=True, cpu_count=16,fit_template=fittemplate,
                                          verbose=False)
        results.check_spectra(path_to_save_figure=os.path.join(Path(__file__).parents[0], "checks2.pdf"),
                              position=((0, 0), (0, 0), (0, 0)))

        s4 = s1.average_spectra_over_region(lonlat_lims=((456 * u.arcsec, 463 * u.arcsec),
                                                         (-47 * u.arcsec, -44.5 * u.arcsec)), allow_reprojection=True)
        results2 = FitResults()

        results2.fit_spice_window_standard(spicewindow=s4, parallelism=False, cpu_count=16,
                                          fit_template=fittemplate, verbose=False)
        results2.check_spectra(path_to_save_figure=os.path.join(Path(__file__).parents[0],
                                                               "checks4.pdf"), position=((0, 0), (0, 0), (0, 0)))
        assert np.abs(results2.components_results["main"]["coeffs"]["I"]["results"].value -
                      results.components_results["main"]["coeffs"]["I"]["results"].value - 0.05324779000000002)   < 0.001

    def test_return_wavelength_array(self, hdu2):
        s1 = SpiceRasterWindowL2(hdu=hdu2)
        lam = s1.return_wavelength_array()

    # s3 = s1.average_spectra_over_region(lonlat_lims=[[]], allo)
