import pytest
from astropy.io import fits
from ..SpiceRasterWindow import SpiceRasterWindowL2
from ..FitResults import FitResults
from ..FittingModel import FittingModel
import os
from pathlib import Path
# This module is used to load images
from PIL import Image
# This module contains a number of arithmetical image operations
from PIL import ImageChops


def image_pixel_differences(base_image, compare_image):
    """
    Calculates the bounding box of the non-zero regions in the image.
    :param base_image: target image to find
    :param compare_image:  set of images containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.

    diff = ImageChops.difference(base_image, compare_image)
    if diff.getbbox():
        return False
    else:
        return True


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

    def test_save_fit_window_standard(self, spicewindow, fittemplate):
        f = FitResults()
        f.fit_spice_window_standard(spicewindow=spicewindow, parallelism=True, cpu_count=16,
                                    fit_template=fittemplate, verbose=False)
        f.to_fits(path_to_save_fits="./core/test/test.fits")
        fig = f.quicklook(show=False)
        fig.savefig("./core/test/test_ql_.png", dpi=50)

        path_fig_ref = "./core/test/test_ql_.png"
        path_fig = "./core/test/test_ql.png"
        base_image = Image.open(path_fig_ref)
        ref_image = Image.open(path_fig)
        assert(image_pixel_differences(base_image, ref_image))

    def test_load_fit_window_standard(self):
        f2 = FitResults()
        f2.from_fits(path_to_fits="./core/test/test.fits")
        fig = f2.quicklook(show=False)
        fig.savefig("./core/test/f2_test_ql.png", dpi=50)

        path_fig_ref = "./core/test/test_ql_.png"
        path_fig = "./core/test/f2_test_ql.png"
        base_image = Image.open(path_fig_ref)
        ref_image = Image.open(path_fig)
        assert(image_pixel_differences(base_image, ref_image))


