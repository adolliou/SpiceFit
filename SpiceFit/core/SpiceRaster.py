import astropy.io.fits
from astropy.io import fits
from yaml import load, Loader
from pathlib import Path
from .SpiceRasterWindow import SpiceRasterWindowL2
import astropy.units as u


class SpiceRaster:
    def __init__(
        self,
        path_l2_fits_file: str = None,
        hdul: astropy.io.fits.HDUList = None,
        windows="all",
    ) -> None:
        """

        :param path_l2_fits_file:
        :param hdul:
        :param windows: "all" : build all windows in file. None : build none. list : build the windows within the list.
        """
        if (path_l2_fits_file is not None) & (hdul is None):
            self.path_l2_fits_file = path_l2_fits_file
            self.hdul = fits.open(path_l2_fits_file)
        elif (path_l2_fits_file is None) & (hdul is not None):
            self.path_l2_fits_file = None
            self.hdul = hdul
        self.wavelength_intervals = None
        self.spectral_windows_names = None
        self.wavelength_intervals = self.get_wavelength_intervals()
        self.spectral_windows_names = self.get_spectral_windows_names()
        list_windows_l2 = self.build_windowsl2(windows)
        self.windows = list_windows_l2
        self.windows_ext = {}
        for win, ext in zip(list_windows_l2, self.spectral_windows_names):
            self.windows_ext[ext] = win

    def get_wavelength_intervals(self) -> list[list[u.Quantity]]:
        """
        get the wavelength intervals corresponding to the hdul windows
        :return:
        """
        wavelength_intervals = []
        hdu = self.hdul[0]
        header = hdu.header
        wavecov_str = header["WAVECOV"]
        wavecov_str = wavecov_str.replace(" ", "")
        wavecov_str_list = wavecov_str.split(",")
        for el in wavecov_str_list:
            el_split = el.split("-")
            wavelength_intervals.append(
                [
                    u.Quantity(float(el_split[0]), "nm"),
                    u.Quantity(float(el_split[1]), "nm"),
                ]
            )
        return wavelength_intervals

    def get_spectral_windows_names(self) -> list[str]:
        """
        get the extension names of the different spectral windows.
        :return:
        """
        spectral_windows_names = []
        for hdu in self.hdul:
            header = hdu.header
            if (header["EXTNAME"] != "VARIABLE_KEYWORDS") and (
                header["EXTNAME"] != "WCSDVARR"
            ):
                spectral_windows_names.append(header["EXTNAME"])
        return spectral_windows_names

    def build_windowsl2(self, windows) -> list:
        """
        build the SpiceRasterWindowL2 elements.
        :param windows:
        :return:
        """
        list_windows = []
        if windows == "all":

            for ii, winname in enumerate(self.spectral_windows_names):
                list_windows.append(SpiceRasterWindowL2(hdu=self.hdul[winname]))
        if type(windows) is list:
            for ii, winname in enumerate(self.spectral_windows_names):
                if (ii in windows) or (winname in windows):
                    list_windows.append(SpiceRasterWindowL2(hdu=self.hdul[winname]))
                else:
                    list_windows.append(None)
        elif windows is None:
            list_windows = [None] * len(self.extension_names)
        return list_windows

    def get_lines_within_wavelength_intervals(
        self, lines_metadata_file: str = None
    ) -> dict:
        """
        :param lines_metadata_file: str, path to a yaml file with personalised lines metadata. If set to none,
        then use the default ones from "metadata_lines_default.yaml"
        """
        lines_in_raster = {}
        if lines_metadata_file is None:
            lines_metadata_file = "../Templates/metadata_lines_default.yaml"
        else:
            lines_metadata_file = Path(lines_metadata_file)
        with open(lines_metadata_file, "r") as f:
            data = load(f, Loader=Loader)
            lines_total = data["list_lines_spice_metadata"]
            for line in lines_total:
                for ii, interval in enumerate(self.wavelength_intervals):
                    if (
                        u.Quantity(line["central_wavelength"], "anstrom") >= interval[0]
                        and u.Quantity(line["central_wavelength"], "angstrom")
                        <= interval[1]
                    ):
                        lines_in_raster[line["name"]] = line
                        lines_in_raster[line["name"]]["window"] = ii
        return lines_in_raster

    def estimate_noise_windows(self, windows="all") -> None:
        if windows == "all":
            for ii, win in enumerate(self.windows):
                win.compute_uncertainty()
        elif type(windows) is list:
            for ii, win in enumerate(self.windows):
                if (ii in windows) or (self.windows_ext[win] in windows):
                    win.compute_uncertainty()
