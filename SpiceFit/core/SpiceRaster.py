import astropy.io.fits
from astropy.io import fits
from yaml import load, Loader
from pathlib import Path
from .SpiceRasterWindow import SpiceRasterWindowL2
import astropy.units as u
from ..util.spice_util import SpiceUtil
import numpy as np
from .FittingModel import FittingModel
from .FitResults import FitResults
import os
import copy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

class SpiceRaster:
    """The Spiceraster class contains all the information about a L2 FITS file of a SPICE raster
    It can detect which lines are present in every window, and fit each window accordingly
    to generate quicklooks showing all of the fitting results. Uses SpiceRasterWindowL2 objects to
    represent the data on each window. 


    """
    sw_wave_limits = u.Quantity([704.0, 790.0], "angstrom")

    def __init__(
        self,
        path_l2_fits_file: str = None,
        hdul: astropy.io.fits.HDUList = None,
        windows="all",
    ) -> None:
        """
        Generate a SpiceRaster object containing all information about a L2 SPICE FITS file.

        :param path_l2_fits_file: path to the SPICE L2 FITS file
        :param hdul: HDUList of the SPICE L2 fits file
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
        
        self.lines = {}
        self.fit_templates = {}
        self.fit_results = {}

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

    def find_lines_in_raster(
            self, 
            lines_metadata_file: str | None = None,
    ):
        """Find all the lines in the raster among those provided with the line_metadata_file. 
        Stores the line results in self.lines , and the FittingModels objects to use for the fitting
        in self.fit_templates 
        If 

        Args:
            lines_metadata_file (str | None, optional): YAML File describing all the lines informations that
            you want to search in the raster. A line not present in the raster will be ignored.
            the lines_metadata_file yaml file can be customized. If set to None, then use the default one in
            the template lines folder  "metadata_lines_default.yaml"
        """        
        
        self.lines = self._get_lines_within_wavelength_intervals(
            lines_metadata_file=lines_metadata_file
            )

        # if the same FittingModel objects are defined, only one remains as it should be to avoir
        # repetitions in the fitting. Also stores the window associated with the template.
        # Can only have one window for one template (no window with overlapping wavelength interval)
        for line in self.lines:
            if line["default_template"] not in self.fit_templates:
                self.fit_templates[line["default_template"]] = {
                    "FittingModel" : FittingModel(filename=line["default_template"]),
                    "window": line["window"]
                }
            elif (self.fit_templates[line["default_template"]]["window"] == line["window"]):
                pass
            else:
                raise ValueError("Two HDUL windows have overlapping wavelength interval")

    def fit_all_windows(self, **kwargs
                       ):
        """Fit all the lines in the raster with the lines in self.lines and the FittingModel objects in self.fitting_model
        store the FitResults objects in a list self.fit_result_list. 
        :param **kwargs : keyword arguments for the FitResults.fit_spice_window_standard function 
        """ 
        if (self.lines == {}) | (self.fit_templates == {}):
            raise ValueError("self.lines or self.fit_templates are empty. Should run self.find_lines_in_raster first.")

        for key in self.fit_templates.keys():
            template_dict = self.fit_templates[key]
            temp = template_dict["FittingModel"]
            win = template_dict["window"]


            res = FitResults()
            res.fit_spice_window_standard(
                spicewindow=self.windows[win],
                fit_template=temp,
                **kwargs,
                )
            self.fit_results[key] = res


    def plot_fittted_map(self, path_to_output_pdf: str, lines="all" | list, **kwargs):
        """Plot results maps with the FitResults.plot_fitted_map function for all the lines
        given as input. The function self.find_lines_in_raster and self.fit_all_windows
        must have been called before. 

        Args:
            path_to_output_pdf: Path to the output multipage PDF figure. Should end with ".pdf".
            lines (_type_, optional): either "all" or a list of the names of the lines you want to plot.
            Defaults to "all" | list[str].
            **kwargs : additional arguments to FitResults.plot_fitted_maps functions
        """
        kwargs["show"] = False
        with PdfPages(filename=path_to_output_pdf) as pdf:
            keys = None
            if lines == "all":
                keys = list(self.lines.keys())
            else:
                keys = lines
            for key in keys:
                res = self.fit_results[key]
                cm = 1/2.56
                fig = plt.figure(figsize=(17 * cm, 17*cm))
                ax = fig.add_subplot()
                res.plot_fitted_map(
                    ax=ax,
                    fig=fig,
                    **kwargs
                )
                pdf.savefig(figure=fig)
                plt.close("all")
                




    def _get_lines_within_wavelength_intervals(
        self, lines_metadata_file: str = None
    ) -> dict:
        """
        :param lines_metadata_file: str, path to a yaml file with personalised lines metadata. If set to none,
        then use the default ones from "metadata_lines_default.yaml"
        """
        lines_in_raster = {}
        if lines_metadata_file is None:
            lines_metadata_file = os.path.join(Path(__file__).parents[0], "Templates/metadata_lines_default.yaml")
        else:
            lines_metadata_file = Path(lines_metadata_file)
        with open(lines_metadata_file, "r") as f:
            data = load(f, Loader=Loader)
            lines_total = data["list_lines_spice_metadata"]
            for line in lines_total:
                for ii, interval in enumerate(self.wavelength_intervals):
                    if (
                        u.Quantity(line["wave"], line["unit"]) >= interval[0]
                        and u.Quantity(line["wave"], line["unit"])
                        <= interval[1]
                    ):
                        lines_in_raster[line["name"]] = line
                        lines_in_raster[line["name"]]["window"] = ii
        return lines_in_raster

    def estimate_noise_windows(self, windows="all") -> None:
        """
        Estimates the noise for the given window of the SpiceRaster object with the sospice package
        """        
        if windows == "all":
            for ii, win in enumerate(self.windows):
                win.compute_uncertainty()
        elif type(windows) is list:
            for ii, win in enumerate(self.windows):
                if (ii in windows) or (self.windows_ext[win] in windows):
                    win.compute_uncertainty()

    def return_wave_calibrated_spice_raster(self, shift_lambda: u.Quantity, detector: int):
        """
        Returns a new SpiceRaster object with a constant wavelength calibration applied to one of the detector

        Args:
            shift_lambda (u.Quantity): The constant shift to apply to one of the detector. The new value will be New = Old + shift
            detector: either "SW" or "LW"
        """        
        list_indexes_window_lw = self._get_windows_lw()
        list_indexes_window_sw = self._get_windows_sw()
        list_to_change = None
        if detector == "LW":
            list_to_change = list_indexes_window_lw
        elif detector == "SW":
            list_to_change = list_indexes_window_sw
        else:
            raise ValueError("detector input must be either LW or SW. ")

        hdul_new = fits.HDUList()
        for ii, win in enumerate(self.windows):
            data  = win.data.copy()
            header = win.header.copy()
            header_new = header.copy()
            if ii in list_to_change:
                header_new["CRVAL3"] = (u.Quantity(header["CRVAL3"], header["CUNIT3"]) + shift_lambda).to(header["CUNIT3"]).value
            else:
                pass
            hdu_new = fits.ImageHDU(data=data, header=header_new)
            hdul_new.append(hdu_new)
        for hdu in self.hdul:
            header = hdu.header
            if (header["EXTNAME"] == "VARIABLE_KEYWORDS") or (
                header["EXTNAME"] == "WCSDVARR"
            ):
                hdu_new =copy.deepcopy(hdu)
                hdul_new.append(hdu_new)
        spiceraster_calibrated = SpiceRaster(hdul=hdul_new)
        return spiceraster_calibrated

    def _get_windows_sw(self):
        """
        returns a list of the windows within the LW detector
        """        
        list_indexes_window_sw = []
        for ii, win in enumerate(self.windows):    
            wave_limits = win.return_wavelength_interval()
            if (wave_limits[0] >= SpiceUtil.get_sw_wave_limits()[0]) & (
                wave_limits[1] <= SpiceUtil.get_sw_wave_limits()[1]
            ):
                list_indexes_window_sw.append(ii)
        return list_indexes_window_sw

    def _get_windows_lw(self):
        """
        returns a list of the windows within the LW detector
        """        
        list_indexes_window_lw = []
        for ii, win in enumerate(self.windows):    
            wave_limits = win.return_wavelength_interval()
            if (wave_limits[0] >= SpiceUtil.get_lw_wave_limits()[0]) & (
                wave_limits[1] <= SpiceUtil.get_lw_wave_limits()[1]
            ):
                list_indexes_window_lw.append(ii)
        return list_indexes_window_lw

    def _find_window_index_from_fittingmodel(self, fitting_model: FittingModel, lines_metadata_file:str|None = None, window_name: str = None):

        # if lines_metadata_file is None:
        #     lines_metadata_file = os.path.join(Path(__file__).parents[0], "Templates/metadata_lines_default.yaml")
        # else:
        #     lines_metadata_file = Path(lines_metadata_file)
        line_name = fitting_model._parinfo["fit"]["main_line"]

        lines_in_raster = self.get_lines_within_wavelength_intervals(lines_metadata_file=lines_metadata_file)
        # find the window where the line exists
        line_in_window = []
        for key in lines_in_raster.keys():
            if line_name.strip() == lines_in_raster[key]["name"]:
                line_in_window.append(lines_in_raster[key]["window"])
        if len(line_in_window) != 1:
            raise ValueError(f"The number of windows where the line is detected should be 1, but is {len(line_in_window)}.\n Either select the window or check the line name")
        window_index = line_in_window[0]
        return window_index


    def __repr__(self):
        s = f"Spiceraster for {self.hdul.info()}"
        return s