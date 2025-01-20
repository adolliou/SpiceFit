from astropy.io import fits
from ..SpiceRaster import SpiceRaster
from matplotlib import pyplot as plt
import numpy as np


def test_SpiceRaster():
    path_fits_l2_spice = ("https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17"
                          "/solo_L2_spice-n-ras_20220317T002536_V03_100663832-017.fits")

    s1 = SpiceRaster(path_l2_fits_file=path_fits_l2_spice)
    with fits.open(path_fits_l2_spice) as hdulist:
        s2 = SpiceRaster(hdul=hdulist)
    s1.estimate_noise_windows()
    assert s1.windows[0].uncertainty is not None
    keys = list(s1.windows_ext.keys())
    assert np.nansum(s1.windows[0].uncertainty["Total"]) == np.nansum(s1.windows_ext[keys[0]].uncertainty["Total"])
    assert np.abs(np.nansum(s1.windows[0].uncertainty["Total"].value) - 16641.86688057675) < 1e-5




