import pytest
from astropy.io import fits
from ..SpiceRasterWindow import SpiceRasterWindowL2
from ..FitResults import FitResults
from ..FittingModel import FittingModel
import os
from pathlib import Path


@pytest.fixture
def hdu():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras"
        "_20220317T002536_V03_100663832-017.fits"
    )  # noqa: E501
    hdu_list = fits.open(url)
    hdu = hdu_list["Ne VIII 770 - Peak"]
    yield hdu
    hdu_list.close()


@pytest.fixture
def spicewindow(hdu):
    return SpiceRasterWindowL2(hdu=hdu)


@pytest.fixture
def fittemplate():
    path_yaml = "./core/test/fit_templates/ne_8_770_42_1c.template.yaml"
    return FittingModel(filename=path_yaml)


class TestFitResults:

    def test_fit_window_standard(self, spicewindow, fittemplate):
        f = FitResults(fit_template=fittemplate, verbose=False)
        f.fit_spice_window_standard(spicewindow=spicewindow, parallelism=True, cpu_count=16, )
        f.to_fits(path_to_save_fits="./core/test/test.fits")



