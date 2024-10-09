import astropy.io.fits
from astropy.io import fits
import astropy.units as u


# f = fits.open("https://spice.osups.universite-paris-saclay.fr/spice-data/release-4.0/level2/2022/03/17/solo_L2_spice-n-ras_20220317T002536_V03_100663832-017.fits")

class SpiceRaster:
    # TODO Only initialize a SpiceRasterWindow when needed.
    def __init__(self, path_l2_fits_file: str = None, hdul: astropy.io.fits.HDUList = None) -> None:
        if (path_l2_fits_file is not None) & (hdul is None):
            self.path_l2_fits_file = path_l2_fits_file
            self.hdul = fits.open(path_l2_fits_file)
        elif (path_l2_fits_file is None) & (hdul is not None):
            self.path_l2_fits_file = None
            self.hdul = hdul
        self.wavelength_intervals = self.get_wavelength_intervals()
        self.extension_names = self.get_extension_names()




    def get_wavelength_intervals(self) -> u.Quantity:
        wavelength_intervals = []
        hdu = self.hdul[0]
        header = hdu.header
        wavecov_str = header["WAVECOV"]
        wavecov_str = wavecov_str.replace(" ", "")
        wavecov_str_list = wavecov_str.split(",")
        for el in wavecov_str_list:
            el_split = el.split("-")
            wavelength_intervals.append([float(el_split[0]), float(el_split[1])])
        wavelength_intervals = u.Quantity(self.wavelength_intervals, "nm")
        return wavelength_intervals

    def get_extension_names(self) -> list[str]:
        extension_names = []
        for hdu in self.hdul:
            header = hdu.header
            extension_names.append(header["EXTNAME"])
        return extension_names

    def get_lines_within_wavelength_intervals(self, lines_metadata_file: str = None) -> dict:
        pass

