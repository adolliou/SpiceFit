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
from matplotlib import pyplot as plt
import astropy.units as u


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
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/"
        "2022/03/17/solo_L2_spice-n-ras"
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
    path_yaml = os.path.join(
        Path(__file__).parents[0], "fit_templates/ne_8_770_42_1c.template.yaml"
    )
    return FittingModel(filename=path_yaml)


@pytest.fixture
def hdu2():
    url = (
        "https://spice.osups.universite-paris-saclay.fr/spice-data/release-5.0/level2/" 
        "2022/03/07/solo_L2_spice-n-ras_20220307T030536_V22_100663723-000.fits"
    )  # noqa: E501
    hdu_list = fits.open(url)
    hdu = hdu_list['C III 977 (Merged)']
    yield hdu
    hdu_list.close()


@pytest.fixture
def spicewindow2(hdu2):
    return SpiceRasterWindowL2(hdu=hdu2)


@pytest.fixture
def fittemplate2():

    return FittingModel(filename="c_3_977_96_1c.template")


class TestFitResults:

    def test_save_fit_window_standard(self, spicewindow, fittemplate):
        path_fig = os.path.join(Path(__file__).parents[0], "test_ql_.png")
        path_fig_ref = os.path.join(Path(__file__).parents[0], "test_ql.png")
        path_fits = os.path.join(Path(__file__).parents[0], "test.fits")

        f = FitResults()
        f.fit_spice_window_standard(
            spicewindow=spicewindow, 
            parallelism=True, 
            cpu_count=16,
            fit_template=fittemplate,
            verbose=2, 
            detrend_doppler=False
            )
        f.to_fits(path_fits)
        fig = f.quicklook(show=False)
        fig.savefig(path_fig, dpi=50)

        base_image = Image.open(path_fig_ref)
        ref_image = Image.open(path_fig)

        assert(image_pixel_differences(base_image, ref_image))

    def test_doppler_detrending(self, spicewindow2, fittemplate2):

        path_fig_doppler = os.path.join(Path(__file__).parents[0], "test_doppler.png")
        path_fig_doppler_ref = os.path.join(Path(__file__).parents[0], "test_doppler_ref.png")

        f = FitResults()
        f.fit_spice_window_standard(
            spicewindow=spicewindow2,
            parallelism=True,
            cpu_count=8,
            fit_template=fittemplate2,
            verbose=2,
            detrend_doppler=True,
        )
        # f.to_fits(os.path.join(Path(__file__).parents[0], "test2.fits"))

        fig = plt.figure()
        ax = fig.add_subplot()
        f.plot_fitted_map(
            fig=fig,
            ax=ax,
            line="main",
            param="delta_x",
            regular_grid=False,
            doppler_mediansubtraction=True,
            imax = 90, 
            sigma_error=1.0,
        )
        fig.savefig(path_fig_doppler, dpi=200)

        base_image = Image.open(path_fig_doppler_ref)
        ref_image = Image.open(path_fig_doppler)

        assert image_pixel_differences(base_image, ref_image)

    def test_save_fit_window_standard_skew(self, spicewindow2, fittemplate2):
        path_fig = os.path.join(Path(__file__).parents[0], "doppler_skew_test.png")
        path_fig_ref = os.path.join(Path(__file__).parents[0], "doppler_skew_ref.png")

        fitres = FitResults()
        fitres.fit_spice_window_skew(
            spicewindow=spicewindow2,
            parallelism=True,
            cpu_count=8,
            fit_template=fittemplate2,
            verbose=False,
            save_folder_skew=Path(__file__).parents[0],
            best_xshift= 2.0,
            best_yshift=-1.667,
            detrend_doppler=True,
        )
        fig = plt.figure()
        ax = fig.add_subplot()
        fitres.plot_fitted_map(
            fig=fig,
            ax=ax,
            line="main",
            param="delta_x",
            regular_grid=False,
            doppler_mediansubtraction=True,
            imax=90,
            sigma_error=2.0,
        )
        fig.savefig(path_fig, dpi=50)
        base_image = Image.open(path_fig_ref)
        ref_image = Image.open(path_fig)
        assert image_pixel_differences(base_image, ref_image)

    def test_load_fit_window_standard(self):
        path_fig = os.path.join(Path(__file__).parents[0], "test_ql_.png")
        path_fig_ref = os.path.join(Path(__file__).parents[0], "test_ql.png")
        path_fits = os.path.join(Path(__file__).parents[0], "test.fits")

        f2 = FitResults()
        f2.from_fits(path_to_fits=path_fits)
        fig = f2.quicklook(show=False)
        fig.savefig(path_fig, dpi=50)

        base_image = Image.open(path_fig_ref)
        ref_image = Image.open(path_fig)
        assert(image_pixel_differences(base_image, ref_image))

    def test_plot_fitted_map(self, hdu):

        path_fits = os.path.join(Path(__file__).parents[0], "test.fits")

        path_fig = os.path.join(Path(__file__).parents[0], "test_fm_.png")
        path_fig2 = os.path.join(Path(__file__).parents[0], "test_fm_2.png")
        path_fig3 = os.path.join(Path(__file__).parents[0], "test_fm_3.png")
        path_fig4 = os.path.join(Path(__file__).parents[0], "test_fm_4.png")

        f2 = FitResults()
        f2.from_fits(path_to_fits=path_fits)
        x = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        y = [[450, 450, 450], [451, 451, 451], [453, 453, 453], [452, 452, 452]]
        pixels = (x, y)

        fig = plt.figure()
        ax=fig.add_subplot()
        f2.plot_fitted_map(fig=fig, ax=ax, line="main", param="radiance",
                           regular_grid=False, pixels=pixels)
        fig.savefig(path_fig2, dpi=50)
        base_image = Image.open(path_fig2)
        ref_image = Image.open(path_fig)
        assert(image_pixel_differences(base_image, ref_image))

        lonlat_lim = ((456 * u.arcsec, 463 * u.arcsec),(-47 * u.arcsec, -44.5 * u.arcsec))
        fig = plt.figure()
        ax=fig.add_subplot()
        f2.plot_fitted_map(fig=fig, ax=ax, line="main", param="radiance",
                           regular_grid=False, lonlat_lims=lonlat_lim, allow_reprojection=True)
        fig.savefig(path_fig4, dpi=50)
        base_image = Image.open(path_fig3)
        ref_image = Image.open(path_fig3)
        assert(image_pixel_differences(base_image, ref_image))

        f2.plot_fitted_map(
            fig=fig,
            ax=ax,
            line="main",
            param="delta_x",
            regular_grid=False,
        )
