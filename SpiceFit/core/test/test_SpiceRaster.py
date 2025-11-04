from astropy.io import fits
from ..SpiceRaster import SpiceRaster
import numpy as np
import pytest
import os
from pathlib import Path
from ..FittingModel import FittingModel
from ..FitResults import FitResults
from ..SpiceRasterWindow import SpiceRasterWindowL2
import astropy.units as u
import astropy.constants as const


@pytest.fixture
def hdul():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/"
        "2022/03/17/solo_L2_spice-n-ras"
        "_20220317T002536_V03_100663832-017.fits"
    )  # noqa: E501
    hdul = fits.open(url)
    return hdul


@pytest.fixture
def fittemplate():
    path_yaml = os.path.join(
        Path(__file__).parents[0], "fit_templates/ne_8_770_42_1c.template.yaml"
    )
    return FittingModel(filename=path_yaml)


class TestSpiceRaster:

    def test_constructor(self):
        path_fits_l2_spice = "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras_20220317T002536_V03_100663832-017.fits"
        s1 = SpiceRaster(path_l2_fits_file=path_fits_l2_spice)
        with fits.open(path_fits_l2_spice) as hdulist:
            s2 = SpiceRaster(hdul=hdulist)
        s1.estimate_noise_windows()
        assert s1.windows[0].uncertainty is not None
        keys = list(s1.windows_ext.keys())
        assert np.nansum(s1.windows[0].uncertainty["Total"]) == np.nansum(
            s1.windows_ext[keys[0]].uncertainty["Total"]
        )
        assert (
            np.abs(np.nansum(s1.windows[0].uncertainty["Total"].to("W / (nm sr m2)").value) - 12163.235854294793)
            < 1e-5
        )

    def test_wavelength_calibration(self, hdul):
        window = "Ly-gamma-CIII group (Merged)"
        velocity_ref = u.Quantity(0, "km/s")

        hdu = hdul[window]
        s = SpiceRasterWindowL2(hdu=hdu)
        fitmod = FittingModel(filename="h_1_972_57_1c.template")

        lambda_ref = u.Quantity(972.57, "angstrom")
        res = FitResults()
        res.fit_spice_window_standard(
            spicewindow=s,
            fit_template=fitmod,
            parallelism=True,
            cpu_count=8,
        )
        res.from_fits(os.path.join(Path(__file__).parents[0], "test2.fits"))
        x_median = np.nanmedian(
            res.components_results["main"]["coeffs"]["x"]["results"]
        )
        x_ref = (lambda_ref * velocity_ref / const.c.to("km/s") ) + lambda_ref

        shift_wave = x_ref - x_median

        raster = SpiceRaster(hdul=hdul)
        raster.return_wave_calibrated_spice_raster(shift_lambda=shift_wave, detector="LW")

