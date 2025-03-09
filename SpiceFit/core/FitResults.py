import string

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LogStretch

from .FittingModel import FittingModel
from .SpiceRasterWindow import SpiceRasterWindowL2
from ..util.shared_memory import gen_shmm
from ..util.fitting import FittingUtil
from ..util.common_util import CommonUtil
from ..util.plotting_fits import PlotFits
import copy
from multiprocessing import Process, Lock
import multiprocessing as mp
import astropy.units as u
import tqdm
from matplotlib import pyplot as plt
from ..util.constants import Constants
from .fitting_spectra import fit_spectra
import matplotlib as mpl
import astropy.constants as const
from matplotlib.gridspec import GridSpec
import random
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import os
import ast

import warnings

warnings.filterwarnings("ignore", message="Card is too long, comment will be truncated.")
warnings.filterwarnings("ignore",
                        message="'UTC' did not parse as fits unit: At col 0, Unit 'UTC'", )


def flatten(xss):
    """
    https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        :param xss:
        :return:
    """
    return [x for xs in xss for x in xs]


class FitResults:

    def __init__(
            self,

    ):
        """
        Initialize a FirResults instance. This class is dedicated to Fit the spectral data on a datacube,
        and to analyze/plot the results.
        """
        self.verbose = None
        self.fit_template = None

        self.fit_function_params = None
        self.fit_function = None

        # data input
        self.data = None  # size : (time, lambda, Y , X)
        self.uncertainty = None  # size : (time, lambda, Y , X)
        self.data_unit = None
        self.lambda_unit = None

        # fitting direct results
        self.fit_results = {}

        # elaborate results
        self.components = None

        # fitting parameters
        self.min_data_points = None
        self.chi2_limit = None

        # dicts to create and gain access to shared memory objects
        self._data_dict = None  # size : (time, lambda, Y , X)
        self._lambda_dict = None  # size : (time, lambda, Y , X)
        self._uncertainty_dict = None  # size : (time, lambda, Y , X)
        self._fit_coeffs_dict = None  # size : (N, time, Y , X)
        self._fit_coeffs_error_dict = None  # size : (N, time, Y , X)
        self._fit_chi2_dict = None  # size : (N, time, Y , X)
        self._flagged_pixels_dict = None  # size : (time, Y , X)

        self.fit_results = None
        self.lock = None
        self.cpu_count = None
        self.display_progress_bar = None
        self.spectral_window = None

        self.components_results = {}
        self.main_line = None

    def fit_spice_window_standard(self,
                                  spicewindow: SpiceRasterWindowL2,
                                  fit_template: FittingModel,
                                  parallelism: bool = True,
                                  cpu_count: int = None,
                                  min_data_points: int = 5,
                                  chi2_limit: float = 20.0,
                                  verbose=0, ):
        """

        Fit all pixels of the field of view for a given SpiceRasterWindowL2 class instance.

        :param verbose:
        :param fit_template: FittingModel object.
        :param spicewindow: SpiceRasterWindowL2 class
        :param parallelism: allow parallelism or not.
        :param cpu_count: cpu counts to use for parallelism.
        :param min_data_points: minimum data points for each pixel with for the fitting.
        :param chi2_limit: limit the chi^2 for a pixel. Above this value, the pixel will be flagged.
        :param display_progress_bar: display the progress bar
        """
        self.verbose = verbose
        self.fit_template = fit_template
        self.fit_results = {
            "coeff": [],  # (N, time, Y , X)
            "chi2": [],  # (time, Y , X)
            "flagged_pixels": [],  # (time, Y , X)
            "name": self.fit_template.params_free["notation"],
        }
        if spicewindow.uncertainty is None:
            spicewindow.compute_uncertainty()

        data_cube = spicewindow.data
        uncertainty_cube = spicewindow.uncertainty["Total"]
        xx, yy, ll, tt = spicewindow.return_point_pixels()
        coords, lambda_cube, t = spicewindow.wcs.pixel_to_world(xx, yy, ll, tt)

        self.fit_window_standard_3d(
            data_cube=data_cube,
            uncertainty_cube=uncertainty_cube,
            lambda_cube=lambda_cube,
            parallelism=parallelism,
            cpu_count=cpu_count,
            min_data_points=min_data_points,
            chi2_limit=chi2_limit,
        )

        self.spectral_window = spicewindow

    def fit_window_standard_3d(
            self,
            data_cube,
            uncertainty_cube,
            lambda_cube,
            parallelism: bool = True,
            cpu_count: int = None,
            min_data_points: int = 5,
            chi2_limit: float = 20.0,
            display_progress_bar=True
    ):
        """
        Fit all pixels of the field of view for a given SpiceRasterWindowL2 class instance. Use this function
        if the data cube has the shape (time, lambda, Y , X)

        :param data_cube:                       (time, lambda, Y , X)
        :param uncertainty_cube:                (time, lambda, Y , X)
        :param lambda_cube:                     (time, lambda, Y , X)
        :param parallelism: allow parallelism or not.
        :param cpu_count: cpu counts to use for parallelism.
        :param min_data_points: minimum data points for each pixel with for the fitting.
        :param chi2_limit: limit the chi^2 for a pixel. Above this value, the pixel will be flagged.
        :param display_progress_bar: display the progress bar
        """
        self.min_data_points = min_data_points
        self.chi2_limit = chi2_limit
        self.display_progress_bar = display_progress_bar

        if cpu_count is None:
            self.cpu_count = mp.cpu_count()
        else:
            self.cpu_count = cpu_count
        # self._gen_function()

        self.lock = Lock()

        lambda_ = lambda_cube.to(Constants.conventional_lambda_units).value
        data_cube = data_cube.to(Constants.conventional_spectral_units).value

        self.lambda_unit = Constants.conventional_lambda_units
        self.data_unit = Constants.conventional_spectral_units

        shmm_data, data = gen_shmm(create=True, ndarray=np.array(data_cube, dtype=np.float64))
        shmm_lambda_, lambda_ = gen_shmm(create=True, ndarray=np.array(lambda_, dtype=np.float64))
        shmm_uncertainty, uncertainty = gen_shmm(
            create=True,
            ndarray=np.array(
                uncertainty_cube
                .to(Constants.conventional_spectral_units)
                .value,
                dtype=np.float64,
            ),
        )
        shmm_fit_coeffs, fit_coeffs = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    np.sum(self.fit_template.parinfo["fit"]["n_components"]),
                    data_cube.shape[0],
                    data_cube.shape[2],
                    data_cube.shape[3],
                ),
                dtype="float",
            ),
        )

        shmm_fit_coeffs_error, fit_coeffs_error = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    np.sum(self.fit_template.parinfo["fit"]["n_components"]),
                    data_cube.shape[0],
                    data_cube.shape[2],
                    data_cube.shape[3],
                ),
                dtype="float",
            ),
        )

        shmm_fit_chi2, fit_chi2 = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    data_cube.shape[0],
                    data_cube.shape[2],
                    data_cube.shape[3],
                ),
                dtype="float",
            ),
        )

        shmm_flagged_pixels, flagged_pixels = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    data_cube.shape[0],
                    data_cube.shape[2],
                    data_cube.shape[3],
                ),
                dtype="bool",
            ),
        )

        self._data_dict = {
            "name": shmm_data.name,
            "dtype": data.dtype,
            "shape": data.shape,
        }
        self._lambda_dict = {
            "name": shmm_lambda_.name,
            "dtype": lambda_.dtype,
            "shape": lambda_.shape,
        }
        self._uncertainty_dict = {
            "name": shmm_uncertainty.name,
            "dtype": uncertainty.dtype,
            "shape": uncertainty.shape,
        }
        self._fit_coeffs_dict = {
            "name": shmm_fit_coeffs.name,
            "dtype": fit_coeffs.dtype,
            "shape": fit_coeffs.shape,
        }
        self._fit_coeffs_error_dict = {
            "name": shmm_fit_coeffs_error.name,
            "dtype": fit_coeffs_error.dtype,
            "shape": fit_coeffs_error.shape,
        }
        self._fit_chi2_dict = {
            "name": shmm_fit_chi2.name,
            "dtype": fit_chi2.dtype,
            "shape": fit_chi2.shape,
        }
        self._flagged_pixels_dict = {
            "name": shmm_flagged_pixels.name,
            "dtype": flagged_pixels.dtype,
            "shape": flagged_pixels.shape,
        }

        self._flag_pixels_with_not_enough_data()

        # Get indexes and positions in (t, i, j) for all pixels of the raster
        t_array, i_array, j_array = np.meshgrid(
            np.arange(data_cube.shape[0]),
            np.arange(data_cube.shape[2]),
            np.arange(data_cube.shape[3]),
            indexing="ij",
        )
        t_array = t_array.flatten()
        i_array = i_array.flatten()
        j_array = j_array.flatten()
        indexes = np.arange(0, i_array.size)

        shmm_flagged_pixels, flagged_pixels = gen_shmm(
            create=False, **self._flagged_pixels_dict
        )
        is_not_flagged = ~flagged_pixels[t_array, i_array, j_array]
        if self.verbose >= 1:
            print(f'Fitting {len(indexes[is_not_flagged])}/{len(indexes)} pixels')

        processes = []
        t_array_to_fit = t_array[is_not_flagged],
        i_array_to_fit = i_array[is_not_flagged],
        j_array_to_fit = j_array[is_not_flagged],
        indexes_to_fit = indexes[is_not_flagged],

        if parallelism:

            t_array_to_fit_sublists = np.array_split(t_array_to_fit[0], self.cpu_count)
            i_array_to_fit_sublists = np.array_split(i_array_to_fit[0], self.cpu_count)
            j_array_to_fit_sublists = np.array_split(j_array_to_fit[0], self.cpu_count)
            indexes_to_fit_sublists = np.array_split(indexes_to_fit[0], self.cpu_count)
            for t_list, i_list, j_list, index_list in zip(t_array_to_fit_sublists,
                                                          i_array_to_fit_sublists,
                                                          j_array_to_fit_sublists,
                                                          indexes_to_fit_sublists):
                kwargs = {"t_list": t_list, "i_list": i_list, "j_list": j_list, "index_list": index_list,
                          "lock": self.lock}

                processes.append(
                    Process(target=self._fit_multiple_pixels_parallelism_3d, kwargs=kwargs)
                )

            lenp = len(processes)
            ii = -1
            is_close = []
            while ii < lenp - 1:
                ii += 1
                processes[ii].start()
                # Wait here as long as the number of alive jobs is superior to the number of counts.

                while (
                        np.sum(
                            [
                                p.is_alive()
                                for mm, p in zip(range(lenp), processes)
                                if (mm not in is_close)
                            ]
                        )
                        > self.cpu_count
                ):
                    pass
                # Close the finished processes before starting a new job to save memory.

                for kk, P in zip(range(lenp), processes):
                    if kk not in is_close:
                        if (not (P.is_alive())) and (kk <= ii):
                            P.close()
                            is_close.append(kk)
            # Waiting for the remaining jobs to complete
            while (
                    np.sum(
                        [
                            p.is_alive()
                            for mm, p in zip(range(lenp), processes)
                            if (mm not in is_close)
                        ]
                    )
                    != 0
            ):
                pass
            # Close the final finished jobs

            for kk, P in zip(range(lenp), processes):
                if kk not in is_close:
                    if (not (P.is_alive())) and (kk <= ii):
                        P.close()
                        is_close.append(kk)

        else:

            self._fit_multiple_pixels_parallelism_3d(
                t_array_to_fit[0],
                i_array_to_fit[0],
                j_array_to_fit[0],
                indexes_to_fit[0],
                self.lock)

        shmm_fit_coeffs_all, fit_coeffs_all = gen_shmm(create=False, **self._fit_coeffs_dict)
        shmm_fit_coeffs_error_all, fit_coeffs_error_all = gen_shmm(create=False, **self._fit_coeffs_error_dict)

        shmm_fit_chi2_all, fit_chit2_all = gen_shmm(create=False, **self._fit_chi2_dict)
        shmm_flagged_pixels, flagged_pixels = gen_shmm(create=False, **self._flagged_pixels_dict)

        coeffs_cp = copy.deepcopy(fit_coeffs_all)
        coeffs_error_cp = copy.deepcopy(fit_coeffs_error_all)

        self.fit_results["unique_index"] = self.fit_template.params_free["unique_index"]
        self.fit_results["coeff_name"] = self.fit_template.params_free["notation"]
        self.fit_results["coeff"] = coeffs_cp
        self.fit_results["coeffs_error"] = coeffs_error_cp
        self.fit_results["chi2"] = copy.deepcopy(fit_chit2_all)
        self.fit_results["flagged_pixels"] = copy.deepcopy(flagged_pixels)
        self.fit_results["unit"] = self.fit_template.params_free["unit"]
        self.fit_results["trans_a"] = self.fit_template.params_free["trans_a"]
        self.fit_results["trans_b"] = self.fit_template.params_free["trans_b"]

        self.build_components_results()

        shmm_data.close()
        shmm_data.unlink()
        shmm_lambda_.close()
        shmm_lambda_.unlink()
        shmm_uncertainty.close()
        shmm_uncertainty.unlink()

        shmm_flagged_pixels.close()
        shmm_flagged_pixels.unlink()
        shmm_fit_chi2_all.close()
        shmm_fit_chi2_all.unlink()
        shmm_fit_coeffs_all.close()
        shmm_fit_coeffs_all.unlink()
        shmm_fit_coeffs_error_all.close()
        shmm_fit_coeffs_error_all.unlink()

    def fit_pixels(self, t_list, i_list, j_list, ModelFit: FittingModel = None,
                   minimum_data_points: int = 5, chi_limit: float = 100):
        """Fit Individual pixels with potentially a new Fitting_model, without parallelism.
        Ideal to compute different fitting with various parameters on a given specific pixel
            the datacube has the form data[t, i, j]
        Args:
            t_list (_type_): list of the t components, normally time
            i_list (_type_): list of the x components
            j_list (_type_): list of the y components
            ModelFit (FittingModel, optional): Fit according to a new fitting model set by the user. Defaults to None.
            minimum_data_points (int, optionql): minimum number of datapoints which are not nan to start the fitting. 
            If below, then the pixel is flagged. 
        """
        if ModelFit is not None:
            fit_template = ModelFit
        else:
            fit_template = self.fit_template

        data_cube = self.spectral_window.data
        uncertainty_cube = self.spectral_window.uncertainty["Total"]
        xx, yy, ll, tt = self.spectral_window.return_point_pixels()
        coords, lambda_cube, t = self.spectral_window.wcs.pixel_to_world(xx, yy, ll, tt)

        lambda_ = np.array(lambda_cube.to(Constants.conventional_lambda_units).value, dtype=np.float64)
        data_cube = np.array(data_cube.to(Constants.conventional_spectral_units).value, dtype=np.float64)
        uncertainty_cube = np.array(uncertainty_cube.to(Constants.conventional_spectral_units).value, dtype=np.float64)

        for t, i, j in tqdm.tqdm(zip(t_list, i_list, j_list), total=len(t_list)):
            x = lambda_[t, :, i, j]
            y = data_cube[t, :, i, j]
            dy = uncertainty_cube[t, :, i, j]

            try:
                popt, pcov = fit_spectra(x=x,
                                         y=y,
                                         dy=dy,
                                         fit_template=fit_template,
                                         minimum_data_points=minimum_data_points)

                chi2 = np.sum(np.diag(pcov))
                if chi2 <= chi_limit:
                    self.fit_results["coeff"][:, t, i, j] = popt
                    self.fit_results["coeffs_error"][:, t, i, j] = np.sqrt(np.diag(pcov))
                    self.fit_results["chi2"][t, i, j] = chi2
                    self.fit_results["flagged_pixels"][t, i, j] = False

                else:
                    self.fit_results["flagged_pixels"][t, i, j] = True
            except ValueError:
                self.fit_results["flagged_pixels"][t, i, j] = True
        self.build_components_results()

    def build_components_results(self):
        type_list, index_list, coeff_list = self.fit_template.gen_mapping_params()

        flagged_pixels = self.fit_results["flagged_pixels"]
        for type_, index_, coeff_ in zip(type_list, index_list, coeff_list):
            a = self.fit_template.params_all[type_][index_][coeff_]
            wha = np.where(a["unique_index"] == np.array(self.fit_results["unique_index"]))[0][0]
            if a["name_component"] not in self.components_results:
                self.components_results[a["name_component"]] = {
                    "info": {
                        "name_component": a["name_component"],
                        "type": type_,
                    },
                    "coeffs": {},
                }
            if a["free"]:
                self.components_results[a["name_component"]]["coeffs"][coeff_] = {
                    "results": u.Quantity(self.fit_results["coeff"][wha, ...], self.fit_results["unit"][wha]),
                    "sigma": u.Quantity(self.fit_results["coeffs_error"][wha, ...], self.fit_results["unit"][wha]),
                    "trans_a": self.fit_results["trans_a"][wha],
                    "trans_b": self.fit_results["trans_b"][wha],
                    "guess": u.Quantity(a["guess"], self.fit_results["unit"][wha]),
                    "max": u.Quantity(a["bounds"][1], self.fit_results["unit"][wha]),
                    "min": u.Quantity(a["bounds"][0], self.fit_results["unit"][wha]),

                    "type": type_,

                }


            else:
                # Not yet impletented the error in the case the parameters are not free
                dict_const = a["type_constrain"]
                b = self.fit_template.gen_coeff_from_unique_index(dict_const["ref"])
                whb = np.where(b["unique_index"] == np.array(self.fit_results["unique_index"]))[0][0]

                self.components_results[a["name_component"]]["coeffs"][coeff_] = {
                    "trans_a": self.fit_results["trans_a"][wha],
                    "trans_b": self.fit_results["trans_b"][wha],
                    "guess": u.Quantity(a["guess"], self.fit_results["unit"][wha]),
                    "max": u.Quantity(a["bounds"][1], self.fit_results["unit"][wha]),
                    "min": u.Quantity(a["bounds"][0], self.fit_results["unit"][wha]),
                }

                if dict_const["operation"] == "plus":
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["results"] = u.Quantity(
                        self.fit_results["coeff"][whb, ...] + \
                        dict_const["value"], self.fit_results["unit"][whb])
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["sigma"] = u.Quantity(
                        self.fit_results["sigma"][whb, ...], self.fit_results["unit"][whb])

                elif dict_const["operation"] == "minus":
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["results"] = u.Quantity(
                        self.fit_results["coeff"][whb, ...] - \
                        dict_const["value"], self.fit_results["unit"][whb])
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["sigma"] = u.Quantity(
                        self.fit_results["sigma"][whb, ...], self.fit_results["unit"][whb])
                elif dict_const["operation"] == "times":
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["results"] = u.Quantity(
                        self.fit_results["coeff"][whb, ...] * \
                        dict_const["value"], self.fit_results["unit"][whb])
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["sigma"] = u.Quantity(
                        self.fit_results["sigma"][whb, ...] * dict_const["value"], self.fit_results["unit"][whb])
                elif dict_const["operation"] == "constant":

                    self.components_results[a["name_component"]]["coeffs"][coeff_]["results"] = u.Quantity(
                        dict_const["value"], self.fit_results["unit"][whb])
                    self.components_results[a["name_component"]]["coeffs"][coeff_]["sigma"] = u.Quantity(
                        0, self.fit_results["unit"][whb])

                else:
                    raise NotImplementedError
                self.components_results[a["name_component"]]["coeffs"][coeff_]["constrain"] = {
                    "reference": dict_const["ref"],
                    "operation": dict_const["operation"],
                    "value": dict_const["value"],
                }
            self.components_results[a["name_component"]]["coeffs"][coeff_]["results"][flagged_pixels] = np.nan
            self.components_results[a["name_component"]]["coeffs"][coeff_]["sigma"][flagged_pixels] = np.nan
        self.components_results["chi2"] = {
            "info": {
                "name_component": "chi2",
                "type": "chi2",
            },
            "coeffs": {
                "chi2": {
                    "results": self.fit_results["chi2"],
                }
            }
        }

        self.components_results["flagged_pixels"] = {
            "info": {
                "name_component": "flagged_pixels",
                "type": "flagged_pixels",
            },
            "coeffs": {
                "flagged_pixels": {
                    "results": self.fit_results["chi2"],
                }
            },
        }
        for type_, index_, coeff_ in zip(type_list, index_list, coeff_list):

            if (type_ == "gaussian") and ("radiance" not in self.components_results[a["name_component"]].keys()):
                a = self.fit_template.params_all[type_][index_][coeff_]

                I = self.components_results[a["name_component"]]["coeffs"]["I"]["results"]
                x = self.components_results[a["name_component"]]["coeffs"]["x"]["results"]
                s = self.components_results[a["name_component"]]["coeffs"]["s"]["results"]

                line = None
                for line_ in self.fit_template.parinfo["info"]:
                    if line_["name"] == a["name_component"]:
                        line = line_
                if line is None:
                    raise NotImplementedError
                lambda_ref = u.Quantity(line["wave"], (line["unit_wave"]))
                self.components_results[a["name_component"]]["coeffs"]["velocity"] = {
                    "results": (const.c.to("km/s") * (x - lambda_ref) / lambda_ref).to(
                        Constants.conventional_velocity_units)
                }
                self.components_results[a["name_component"]]["coeffs"]["radiance"] = {
                    "results": (I * np.sqrt(2 * np.pi * s * s)).to(Constants.conventional_radiance_units)
                }
                self.components_results[a["name_component"]]["coeffs"]["fwhm"] = {
                    "results": 2.355 * s
                }
                self.components_results[a["name_component"]]["coeffs"]["velocity"]["results"][flagged_pixels] = np.nan
                self.components_results[a["name_component"]]["coeffs"]["radiance"]["results"][flagged_pixels] = np.nan
                self.components_results[a["name_component"]]["coeffs"]["fwhm"]["results"][flagged_pixels] = np.nan
                self.components_results[a["name_component"]]["coeffs"]["I"]["results"][flagged_pixels] = np.nan
                self.components_results[a["name_component"]]["coeffs"]["x"]["results"][flagged_pixels] = np.nan
                self.components_results[a["name_component"]]["coeffs"]["s"]["results"][flagged_pixels] = np.nan
        dic = copy.deepcopy(self.components_results)

        for line_ in dic.keys():
            if line_ == self.fit_template.parinfo["main_line"]:
                self.components_results["main"] = self.components_results[line_]
        self.main_line = self.fit_template.parinfo["main_line"]

    def _fit_multiple_pixels_parallelism_3d(self, t_list, i_list, j_list, index_list, lock):

        shmm_data_all, data_all = gen_shmm(create=False, **self._data_dict)
        shmm_lambda_all, lambda_all = gen_shmm(create=False, **self._lambda_dict)
        shmm_uncertainty_all, uncertainty_all = gen_shmm(
            create=False, **self._uncertainty_dict
        )
        shmm_fit_coeffs_all, fit_coeffs_all = gen_shmm(
            create=False, **self._fit_coeffs_dict
        )
        shmm_fit_coeffs_all_error, fit_coeffs_all_error = gen_shmm(
            create=False, **self._fit_coeffs_error_dict
        )
        shmm_fit_chi2_all, fit_chit2_all = gen_shmm(create=False, **self._fit_chi2_dict)
        shmm_flagged_pixels, flagged_pixels = gen_shmm(
            create=False, **self._flagged_pixels_dict
        )

        if self.verbose >= 1:
            for t, i, j, index in tqdm.tqdm(zip(t_list, i_list, j_list, index_list), total=len(t_list)):
                x = lambda_all[t, :, i, j]
                y = data_all[t, :, i, j]
                dy = uncertainty_all[t, :, i, j]
                try:
                    popt, pcov = fit_spectra(x=x,
                                             y=y,
                                             dy=dy,
                                             fit_template=self.fit_template,
                                             minimum_data_points=self.min_data_points)

                    chi2 = np.sum(np.diag(pcov))
                    if chi2 <= self.chi2_limit:
                        lock.acquire()
                        fit_coeffs_all[:, t, i, j] = popt
                        fit_coeffs_all_error[:, t, i, j] = np.sqrt(np.diag(pcov))
                        fit_chit2_all[t, i, j] = chi2
                        lock.release()
                    else:
                        lock.acquire()
                        fit_coeffs_all[:, t, i, j] = popt
                        fit_coeffs_all_error[:, t, i, j] = np.sqrt(np.diag(pcov))
                        fit_chit2_all[t, i, j] = chi2
                        flagged_pixels[t, i, j] = True
                        lock.release()
                except ValueError:
                    lock.acquire()
                    flagged_pixels[t, i, j] = True
                    lock.release()

        else:
            for t, i, j, index in zip(t_list, i_list, j_list, index_list):
                x = lambda_all[t, :, i, j]
                y = data_all[t, :, i, j]
                dy = uncertainty_all[t, :, i, j]
                try:
                    popt, pcov = fit_spectra(x=x,
                                             y=y,
                                             dy=dy,
                                             fit_template=self.fit_template,
                                             minimum_data_points=self.min_data_points)
                    if popt is not None:

                        chi2 = np.sum(np.diag(pcov))
                        if chi2 <= self.chi2_limit:
                            lock.acquire()
                            fit_coeffs_all[:, t, i, j] = popt
                            fit_coeffs_all_error[:, t, i, j] = np.sqrt(np.diag(pcov))
                            fit_chit2_all[t, i, j] = chi2
                            lock.release()
                        else:
                            lock.acquire()
                            flagged_pixels[t, i, j] = True
                            lock.release()
                    else:
                        lock.acquire()
                        flagged_pixels[t, i, j] = True
                        lock.release()
                except ValueError:
                    lock.acquire()
                    flagged_pixels[t, i, j] = True
                    lock.release()

        shmm_data_all.close()
        shmm_lambda_all.close()
        shmm_uncertainty_all.close()
        shmm_fit_coeffs_all.close()
        shmm_fit_chi2_all.close()
        shmm_flagged_pixels.close()
        shmm_fit_coeffs_all_error.close()

    def _gen_function(self):
        type = self.fit_template.parinfo["fit"]["type"]
        only_gaussi_const = np.logical_and(
            all([n in ["gaussian", "polynomial"] for n in type]),
            len(np.array(type)[np.array(type) == "const"]) == 1,
        )
        if only_gaussi_const:
            self.fit_function = self._gen_function_only_gaussians()
        else:
            self.fit_function = self._gen_function_exec()

    def _gen_function_only_gaussians(self):
        type = self.fit_template.parinfo["fit"]["type"]
        # Set the constant at the -1 index
        index_background = np.where(np.array(type) == "const")[0][0]
        if index_background != len(type) - 1:
            for key in self.fit_template.parinfo["fit"].keys():
                back_value = copy.deepcopy(
                    self.fit_template.parinfo["fit"][key][index_background]
                )
                self.fit_template.parinfo["fit"][key].pop(index_background)
                self.fit_template.parinfo["fit"][key].append(back_value)
        return FittingUtil.multiple_gaussian_cfit

    def _gen_function_exec(self):
        """
        Create a complex function by writing a str, then running this str with the exec method to execute it.
        """
        raise NotImplementedError


    def _flag_pixels_with_not_enough_data(self):

        shmm_data, data = gen_shmm(create=False, **self._data_dict)
        shmm_flagged_pixels, flagged_pixels = gen_shmm(
            create=False, **self._flagged_pixels_dict
        )
        flagged_pixels[...] = (~np.isnan(data)).sum(axis=1) < self.min_data_points

        shmm_data.close()
        shmm_flagged_pixels.close()

    @staticmethod
    def _transform_to_conventional_unit(quantity: u.Quantity) -> u.Quantity:
        """
        transform an u.quantity into either a nm or a W/ (m2 sr nm), which are the conventional units for the fitting.
        :param quantity:
        """
        if quantity.unit.is_equivalent(Constants.conventional_lambda_units):
            return quantity.to(Constants.conventional_lambda_units)
        elif quantity.unit.is_equivalent(Constants.conventional_spectral_units):
            return quantity.to(Constants.conventional_spectral_units)
        else:
            raise ValueError(f"Cannot convert {quantity} to conventional unit")

    def quicklook_coefficient(self, coeff_index: int, fig=None, ax=None):
        """
        Plot a quicklook plot of the given coefficient index
        :param coeff_index: coefficient index to plot
        :param fig:
        :param ax:
        """
        if self.spectral_window is None:
            raise ValueError("The data is still not fitted")

        w_xy = self.spectral_window.w_xy

        # x, y = np.meshgrid(np.arange(self.spectral_window.data.shape[3]), np.arange(self.spectral_window.data.shape[2]))
        # coords = w_xy.pixel_to_world(x, y)
        # long, latg, dlon, dlat = PlotFits.build_regular_grid(coords.Tx, coords.Ty)
        # long_arc = CommonUtil.ang2pipi(long.to("arcsec")).value
        # latg_arc = CommonUtil.ang2pipi(latg.to("arcsec")).value
        # dlon = dlon.to("arcsec").value
        # dlat = dlat.to("arcsec").value

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
        cmap.set_bad('white')
        coeff = self.fit_results["coeff"][coeff_index, 0, :, :]
        cdelt1 = self.spectral_window.header["CDELT1"]
        cdelt2 = self.spectral_window.header["CDELT2"]
        ratio = cdelt2 / cdelt1
        im = ax.imshow(coeff, origin="lower", interpolation="none",
                       cmap=cmap, aspect=ratio)

        fig.colorbar(im, ax=ax)

        return fig

    def quicklook(self, line="main", show=True):
        """
        Function to quickly plot the main results of a fitted line
        :param line: line to plot the results
        """

        if line not in self.components_results.keys():
            raise ValueError(f"Cannot plot {line} as it is not a fitted line")

        cm = Constants.inch_to_cm
        fig = plt.figure(figsize=(17 * cm, 17 * cm))
        gs = GridSpec(2, 2, wspace=0.3, hspace=0.3)
        axs = [fig.add_subplot(gs[i, j]) for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1])]
        for ii, param in enumerate(["radiance", "velocity", "fwhm", "chi2"]):
            ax = axs[ii]
            self.plot_fitted_map(ax, fig, line, param)
        if show == True:
            fig.show()
        else:
            return fig

    def plot_fitted_map(self, ax, fig, line, param, regular_grid=False,
                        coords: SkyCoord = None, lonlat_lims: tuple = None, pixels: tuple = None,
                        allow_reprojection: bool = False
                        ):
        """

        Plot fitted parameters (such as radiance of a given map.)
        :param ax: ax of the image.
        :param fig: figure object
        :param line: line name. write "main" for the main line of the template.
        :param param: parameter to plot, between radiance,  fwhm, velocity and chi2.
        :param regular_grid:

        The following parameters are used if one want to show a sub fov.
         Please refer to RasterWindow.average_spectra_over_region for a detailed description of the parameters
        :param coords:
        :param pixels:
        :param lonlat_lims:
        :param allow_reprojection:
        """
        units = {
            "radiance": Constants.conventional_radiance_units,
            "fwhm": Constants.conventional_lambda_units,
            "velocity": Constants.conventional_velocity_units,
            "chi2": None
        }
        cmaps = {
            "radiance": mpl.colormaps.get_cmap('viridis'),
            "fwhm": mpl.colormaps.get_cmap('viridis'),
            "velocity": mpl.colormaps.get_cmap('bwr'),
            "chi2": mpl.colormaps.get_cmap('viridis'),
        }

        unit = units[param]
        cmap = cmaps[param]  # viridis is the default colormap for imshow
        cmap.set_bad('white')
        a = self.components_results[line]["coeffs"]
        w_xy = self.spectral_window.w_xy

        if param == "chi2":
            data = self.components_results["chi2"]["coeffs"]["chi2"]["results"]
        else:
            data = a[param]["results"].to(unit).value
            x, y = np.meshgrid(np.arange(self.spectral_window.data.shape[2]),
                               np.arange(self.spectral_window.data.shape[1]))
        if data.ndim == 3:
            data = data[0, ...]
            x, y = np.meshgrid(np.arange(self.spectral_window.data.shape[3]),
                               np.arange(self.spectral_window.data.shape[2]))
        # isnotnan = np.logical_not(np.isnan(data))
        # min_ = np.percentile(data[isnotnan], 1)
        # max_ = np.percentile(data[isnotnan], 100)
        # norm_ = ImageNormalize(stretch=LogStretch(a=30))
        norms = {
            "radiance": PlotFits.get_range(data, stre="log", imin=0, imax=100),
            # "radiance": norm_,
            "fwhm": PlotFits.get_range(data, stre=None),
            "velocity": mpl.colors.CenteredNorm(vcenter=0),
            "chi2": PlotFits.get_range(data, stre=None),
        }

        norm = norms[param]

        if regular_grid:
            coords = w_xy.pixel_to_world(x, y)
            long, latg, dlon, dlat = PlotFits.build_regular_grid(coords.Tx, coords.Ty)
            long_arc = CommonUtil.ang2pipi(long.to("arcsec")).value
            latg_arc = CommonUtil.ang2pipi(latg.to("arcsec")).value
            dlon = dlon.to("arcsec").value
            dlat = dlat.to("arcsec").value
            coordsg = SkyCoord(long, latg, frame=coords.frame)
            xg, yg = w_xy.world_to_pixel(coordsg)
            data_rep = CommonUtil.interpol2d(data, x=xg, y=yg, order=3, fill=np.nan)

            im = ax.imshow(data_rep, origin="lower", interpolation="none", cmap=cmap,
                           extent=(long_arc[0, 0] - 0.5 * dlon, long_arc[-1, -1] + 0.5 * dlon,
                                   latg_arc[0, 0] - 0.5 * dlat, latg_arc[-1, -1] + 0.5 * dlat),
                           norm=norm, )
        else:
            cdelt1 = self.spectral_window.header["CDELT1"]
            cdelt2 = self.spectral_window.header["CDELT2"]
            ratio = cdelt2 / cdelt1

            im = ax.imshow(data, origin="lower", interpolation="none", cmap=cmap,
                           norm=norm, aspect=ratio)

        cbar = fig.colorbar(im, ax=ax, label=unit, pad=0)

        ax.set_xlabel("Solar-X")
        ax.set_ylabel("Solar-Y")
        ax.set_title(param)

        if (coords is not None) or (lonlat_lims is not None) or (pixels is not None):
            (lat_subfov, lon_subfov,
             x_subfov, y_subfov) = self.spectral_window.extract_subfield_coordinates(coords,lonlat_lims,
                                                                                    pixels,
                                                                                    allow_reprojection, )
            if regular_grid:
                lon_subfov_arcsec = lon_subfov.to("arcsec").value
                lat_subfov_arcsec = lat_subfov.to("arcsec").value

                ax.plot(lon_subfov_arcsec, lat_subfov_arcsec, '+', ms=0.7, mew=0.5, c="r")

            else:
                ax.plot(x_subfov, y_subfov, '+', ms=0.7, mew=0.5, c="r")

    def check_spectra(self, path_to_save_figure: str, position="random"):
        """
        show example of the spectra with the fitting.
        :param path_to_save_figure: path where to save the PDF figure. Must end by "pdf".
        :param position: "random" or 2-tuple with the positions of the pixel where the spectra is shown
        ex : position = ( (0, 0, 0), (0, 1, 2) , (0, 0, 0) ) shows the spectrum at data[0, 0, 0] ; data[0, 1, 0]... etc
        """
        x, y = np.meshgrid(np.arange(self.spectral_window.data.shape[3]), np.arange(self.spectral_window.data.shape[2]))
        xf = x.flatten()
        yf = y.flatten()

        xpos = []
        ypos = []
        index = np.arange(len(xf))
        cm = Constants.inch_to_cm
        cdelt1 = self.spectral_window.header["CDELT1"]
        cdelt2 = self.spectral_window.header["CDELT2"]
        ratio = cdelt2 / cdelt1
        if position == "random":
            for ii in range(50):
                index_r = random.choice(index)
                xpos.append(xf[index_r])
                ypos.append(yf[index_r])
        elif isinstance(position, tuple):
            xpos = position[0]
            ypos = position[1]
            if isinstance(xpos, int):
                xpos = [xpos]
                ypos = [ypos]

        if self.spectral_window is None:
            raise NotImplementedError
        data_cube = self.spectral_window.data
        uncertainty_cube = self.spectral_window.uncertainty["Total"]
        xx, yy, ll, tt = self.spectral_window.return_point_pixels()
        coords, lambda_cube, t = self.spectral_window.wcs.pixel_to_world(xx, yy, ll, tt)

        data_l2 = data_cube.to(Constants.conventional_spectral_units).value
        uncertainty_l2 = uncertainty_cube.to(Constants.conventional_spectral_units).value
        lambda_l2 = lambda_cube.to(Constants.conventional_lambda_units).value

        if data_cube.ndim == 4:
            data_l2 = data_l2[0, ...]
            uncertainty_l2 = uncertainty_l2[0, ...]
            lambda_l2 = lambda_l2[0, ...]

        with PdfPages(path_to_save_figure) as pdf:
            for ii in range(len(xpos)):

                fig = plt.figure(figsize=(17 * cm, 9 * cm))
                gs = GridSpec(1, 2, wspace=0.7, hspace=0.7, width_ratios=[0.5, 1.0])
                axs = [fig.add_subplot(gs[n]) for n in range(2)]

                unit = Constants.conventional_radiance_units
                param = "radiance"

                data_radiance = self.components_results["main"]["coeffs"][param]["results"].to(unit).value
                if data_radiance.ndim == 3:
                    data_radiance = data_radiance[0, ...]
                cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
                cmap.set_bad('white')
                im = axs[0].imshow(data_radiance, origin="lower", interpolation="none", cmap=cmap,
                                   aspect=ratio)
                cbar = fig.colorbar(im, ax=axs[0], label=unit, fraction=0.046, pad=0.04)
                axs[0].set_label(param)

                rect = Rectangle((xpos[ii] - 0.5, ypos[ii] - 0.5,), height=1, width=1, linewidth=0.7,
                                 edgecolor='r', facecolor="none")
                axs[0].add_patch(rect)
                axs[1].errorbar(x=lambda_l2[:, ypos[ii], xpos[ii]], y=data_l2[:, ypos[ii], xpos[ii]],
                                yerr=0.5 * uncertainty_l2[:, ypos[ii], xpos[ii]],
                                lw=0.9, marker='', linestyle='-', elinewidth=0.4, color="k", label="data", )
                yfit_total = self.get_fitted_spectra(x=u.Quantity(lambda_l2[:, ypos[ii], xpos[ii]],
                                                                  Constants.conventional_lambda_units),
                                                     position=(0, ypos[ii], xpos[ii]),
                                                     component="total")
                axs[1].plot(lambda_l2[:, ypos[ii], xpos[ii]], yfit_total,
                            lw=0.9, marker='', linestyle='-', color="b", label="total",
                            )

                axs[1].set_xlabel(f"Wavelength [{Constants.conventional_lambda_units}]")
                axs[1].set_ylabel(f"Spectra [{Constants.conventional_spectral_units}]")
                title = f"{str(xpos[ii]), str(ypos[ii])}, "
                for param in self.components_results["main"]["coeffs"].keys():
                    if (param != "radiance") and (param != "velocity") and (param != "s"):
                        title += f"{param} : {self.components_results['main']['coeffs'][param]['results'].value[0,
                        ypos[ii], xpos[ii]]:.2f}, "
                axs[1].set_title(title)

                pdf.savefig(fig)
                plt.close("all")

    def get_fitted_spectra(self, x: np.array, position: tuple, component: str = "total"):
        """
        return the fitted y values for a given wavelength array "x", given a component name. accepts also "all"
        to take all the components
        :param x:
        :param position:
        :param component:
        :return:
        """
        x = x.to(Constants.conventional_lambda_units).value
        if component == "total":
            coeffs = self.fit_results["coeff"]
            if coeffs.ndim == 4:
                coeff = coeffs[:, position[0], position[1], position[2]]
            elif coeffs.ndim == 3:
                coeff = coeffs[:, position[0], position[1]]
            else:
                raise NotImplementedError

            return self.fit_template.fitting_function(x, *coeff)
        else:
            comp = self.components_results[component]
            if comp["info"]["type"] == "gaussian":
                I = comp["coeffs"]["I"]["results"].value[position]
                x = comp["coeffs"]["x"]["results"].value[position]
                s = comp["coeffs"]["s"]["results"].value[position]

                return FittingUtil.gaussian(x, I, x, s, 0)
            elif comp["info"]["type"] == "polynomial":
                value = []
                for letter in string.ascii_lowercase:
                    if letter in comp.keys():
                        value.append(comp[letter])
                return FittingUtil.polynomial(x, *value)
            else:
                raise NotImplementedError

    def to_fits(self, path_to_save_fits: str = None, folder_to_save_fits: str = None, hdu_wcsdvar=None):
        """
        Save the basic fitting data into a FITS file that resembles the ones produced by OSLO.
        The hdulist has between the following windows
        - hdul[0] :  coeffs
        - hdul[1] :  coeffs_sigma
        - hdul[2] :  data_l2
        - hdul[3] : table records that records the fitting model informatin as the _parinfo object.
        - hdul[4] : table with the self.fits_results dictionnary, that greatly facilitates the reconstruction
        of the FitResults object.
        can be used to reconstruct the fitstemplate object.
        if hdu_wcsdvar is not None
        - hdul[5] :  data_l2

        :param hdu_wcsdvar: an additional hdu that can be added at the end of the hdulist to correct the jitter
        information
        :param path_to_save_fits: path where the FITS file to be saved. default to None if folder_to_save_fits is set instead 
        :param folder_to_save_fits: folder where to save the FITS file. The name of the output FITS file will be
        the same as the input L2 file, but with L3 changed into L2 in the filename
        """

        if ((path_to_save_fits is None) and (folder_to_save_fits is None)) or (
                (path_to_save_fits is not None) and (folder_to_save_fits is not None)):
            raise ValueError(" Set only one of those arguments to a str : path_to_save_fits or folder_to_save_fits")

        header_ref = self.spectral_window.header
        hdu = fits.PrimaryHDU()

        if path_to_save_fits is not None:
            path_fits = path_to_save_fits
            filename = os.path.basename(path_to_save_fits)
        elif folder_to_save_fits is not None:
            filename = header_ref["filename"].replace("L2", "L3")
            path_fits = os.path.join(folder_to_save_fits, filename)
        else:
            raise ValueError(" Set only one of those arguments to a str : path_to_save_fits or folder_to_save_fits")
        keys_comp = list(self.components_results.keys())
        keys_comp.remove("main")
        keys_comp.remove("flagged_pixels")
        ncoeff = 0
        for ii, key in enumerate(keys_comp):
            if self.components_results[key]["info"]["type"] == "gaussian":
                ncoeff += 3
            elif self.components_results[key]["info"]["type"] == "polynomial":
                a = self.components_results[key]["coeffs"]
                ncoef = len(a.keys())
                ncoeff += ncoef
            elif self.components_results[key]["info"]["type"] == "chi2":
                ncoeff += 1

        data = self._write_hdu(hdu, header_ref, path_fits, keys_comp=keys_comp, results_type="results",
                               hdu_wcsdvar=hdu_wcsdvar, ncoeff=ncoeff)
        hdu.data = data
        hdu.add_checksum()

        hdu_sigma = fits.ImageHDU()
        data_sigma = self._write_hdu(hdu_sigma, header_ref, path_fits, keys_comp=keys_comp,
                                     results_type="sigma", hdu_wcsdvar=hdu_wcsdvar, ncoeff=ncoeff)

        hdu_sigma.data = data_sigma
        hdu_sigma.add_checksum()

        hdu_data = fits.ImageHDU()
        hdu_data.data = self.spectral_window.data.copy()
        hdu_data.data = hdu_data.data.to(self.spectral_window.header["BUNIT"]).value
        for key in self.spectral_window.header.copy():
            if key != '' and key != 'COMMENT' and key != "HISTORY":
                hdu_data.header[key] = self.spectral_window.header[key]
        if 'XTENSION' not in hdu_data.header:
            hdu_data.header.insert('SIMPLE', ('XTENSION', 'IMAGE   '))
        # hdu_data.header["XTENSION"] = 'IMAGE   '
        # hdu_data.header["PCOUNT"] = 0
        # hdu_data.header["GCOUNT"] = 1
        # hdu_data.header["PCOUNT"] = 0

        hdu_data.header["EXTNAME"] = (f'{header_ref["EXTNAME"]} data', 'Extension name of this window')
        hdu_data.header["FILENAME"] = filename
        hdu_data.header["ORNAME"] = (header_ref["EXTNAME"], 'Original Extension name')
        hdu_data.header['RESEXT'] = (f'{header_ref["EXTNAME"]} results', 'Extension name of results')
        hdu_data.header['UNCEXT'] = (f'{header_ref["EXTNAME"]} sigma', 'Extension name of uncertainties')
        hdu_data.header['DATAEXT'] = (f'{header_ref["EXTNAME"]} data', 'Extension name of data')
        hdu_data.header['PAREXT'] = (f'{header_ref["EXTNAME"]} parinfo', 'Extension name of data')
        hdu_data.header['RECEXT'] = (f'{header_ref["EXTNAME"]} reconstruction', 'Extension name of data')

        if hdu_wcsdvar is not None:
            hdu_data.header['WCSEXT'] = (f'{header_ref["EXTNAME"]} WCSDVARR', 'Extension name of WCSDVARR')

        hdu_parinfo = fits.TableHDU(name=f"{header_ref["EXTNAME"]} parinfo")
        dict_str = str(self.fit_template._parinfo)
        arr = np.array([(dict_str)], dtype=[('parinfo', f'S{len(dict_str)}')])
        hdu_parinfo.data = arr

        hdu_fits_results = fits.ImageHDU(name=f'{header_ref["EXTNAME"]} reconstruction')

        # bytes = arr['parinfo'][0]
        # sss = bytes.decode("UTF-8")
        # ddd = ast.literal_eval(sss)

        rec_coeffs = copy.deepcopy(self.fit_results["coeff"])
        rec_chi2 = copy.deepcopy(self.fit_results["chi2"])
        rec_chi2 = rec_chi2.reshape(1, *rec_chi2.shape)
        rec_coeffs_error = copy.deepcopy(self.fit_results["coeffs_error"])
        flagged = np.zeros_like(self.fit_results["flagged_pixels"])
        flagged[self.fit_results["flagged_pixels"]] = 32000
        flagged[np.logical_not(self.fit_results["flagged_pixels"])] = 0
        flagged = flagged.reshape(1, *flagged.shape)

        data_rec = np.append(rec_coeffs, rec_coeffs_error, axis=0)
        data_rec = np.append(data_rec, rec_chi2, axis=0)
        data_rec = np.append(data_rec, flagged, axis=0)
        hdu_fits_results.data = data_rec

        unique_index = self.fit_results["unique_index"]
        name = self.fit_results["name"]
        coeff_name = self.fit_results["coeff_name"]
        unit = self.fit_results["unit"]
        trans_a = self.fit_results["trans_a"]
        trans_b = self.fit_results["trans_b"]
        hdu_fits_results.header["NCOEFF"] = len(unique_index)
        for ii in range(len(unique_index)):
            hdu_fits_results.header[f"CNAME{ii}"] = name[ii]
            hdu_fits_results.header[f"CUI{ii}"] = unique_index[ii]
            hdu_fits_results.header[f"CCNAME{ii}"] = coeff_name[ii]
            hdu_fits_results.header[f"CUNIT{ii}"] = unit[ii]
            hdu_fits_results.header[f"CTRA{ii}"] = trans_a[ii]
            hdu_fits_results.header[f"CTRB{ii}"] = trans_b[ii]

        hdul = fits.HDUList(hdus=[hdu, hdu_sigma, hdu_data, hdu_parinfo, hdu_fits_results])

        if hdu_wcsdvar is not None:
            hdu_wcs = fits.ImageHDU()
            hdu_wcs.data = hdu_wcsdvar.data.copy()
            hdu_wcs.header = hdu_wcsdvar.header.copy()

            hdu_wcs.header["EXTNAME"] = (f'{header_ref["EXTNAME"]} WCSDVARR', 'Extension name of this window')
            hdu_wcs.header["FILENAME"] = filename
            hdu_wcs.header['ANA_NCMP'] = (len(keys_comp), 'Number of fit components')
            hdu_wcs.header['RESEXT'] = (f'{header_ref["EXTNAME"]} results', 'Extension name of results')
            hdu_wcs.header['UNCEXT'] = (f'{header_ref["EXTNAME"]} sigma', 'Extension name of uncertainties')
            hdu_wcs.header['DATAEXT'] = (f'{header_ref["EXTNAME"]} data', 'Extension name of data')
            hdu_wcs.header['WCSEXT'] = (f'{header_ref["EXTNAME"]} WCSDVARR', 'Extension name of WCSDVARR')
            hdul.append(hdu_wcs)
        hdul.writeto(path_fits, overwrite=True)

    def _write_hdu(self, hdu, header_ref, filename, ncoeff, keys_comp, results_type="results", hdu_wcsdvar=None):
        date_now = Time(datetime.now())
        hdu.header["DATE"] = date_now.fits
        hdu.header["ORNAME"] = (header_ref["EXTNAME"], 'Original Extension name')
        hdu.header["EXTNAME"] = (f'{header_ref["EXTNAME"]} {results_type}', 'Extension name of this window')
        hdu.header["LONGSTRN"] = header_ref["LONGSTRN"]
        hdu.header["FILENAME"] = filename
        hdu.header['ANA_NCMP'] = (len(keys_comp), 'Number of fit components')
        hdu.header['RESEXT'] = (f'{header_ref["EXTNAME"]} results', 'Extension name of results')
        hdu.header['UNCEXT'] = (f'{header_ref["EXTNAME"]} sigma', 'Extension name of uncertainties')
        hdu.header['DATAEXT'] = (f'{header_ref["EXTNAME"]} data', 'Extension name of data')
        hdu.header['PAREXT'] = (f'{header_ref["EXTNAME"]} parinfo', 'Extension name of data')
        hdu.header['RECEXT'] = (f'{header_ref["EXTNAME"]} reconstruction', 'Extension name of data')
        if hdu_wcsdvar is not None:
            hdu.header['WCSEXT'] = (f'{header_ref["EXTNAME"]} WCSDVARR', 'Extension name of data')

        shape = self.fit_results["coeff"][0, ...].shape

        index = 0
        last_index = ['DATAEXT']

        data = np.zeros((ncoeff, *shape), dtype=float)

        for ii, key in enumerate(keys_comp):

            a = self.components_results[key]["coeffs"]
            subkeys = list(a.keys())
            last_index_ = None
            if self.components_results[key]["info"]["type"] == "gaussian":
                hdu.header[f'CMPTYP{str(ii + 1)}'] = ("Gaussian", f"Type of fit component {ii + 1}")
                hdu.header[f'CMPNAM{ii + 1}'] = (key, f'Name of fit component {ii + 1}')
                hdu.header[f'CMP_NP{ii + 1}'] = (len(subkeys), f'Number of parameters in fit component {ii + 1}')

                for param, letter, name, transa, transb in zip(["I", "x", "s"], ["A", "B", "C"],
                                                               ["Amplitude", "Position", "Width"],
                                                               [1, 1, 0.424661], [0, 0, 0]):
                    data[index, ...] = self.components_results[key]["coeffs"][param][results_type]
                    index = index + 1
                    hdu.header[f'PNAME{ii + 1}{letter}'] = (name, f'Name of parameter {name} '
                                                                  f'for component {ii + 1}')
                    hdu.header[f'PUNIT{ii + 1}{letter}'] = (str(a[param]['results'].unit),
                                                            f'Phys. unit of parameter {name} '
                                                            f'for component {ii + 1}')
                    hdu.header[f'PDESC{ii + 1}{letter}'] = (f'This parameter describes the {name} of the '
                                                            f'Gaussian, in the'
                                                            'same units as the data being fitted',
                                                            f'Description of '
                                                            f'parameter a for '
                                                            f'component {ii + 1}')
                    unit = a[param]['results'].unit
                    hdu.header[f'PINIT{ii + 1}{letter}'] = (
                        (a[param]["guess"].to(unit).value - transb) / transa,
                        f'Initial value of parameter {name} '
                        f'for component {ii + 1}')
                    hdu.header[f'PMAX{ii + 1}{letter}'] = ((a[param]["max"].to(unit).value - transb) / transa,
                                                           f'Maximum value of parameter {name} '
                                                           f'for component {ii + 1}')
                    hdu.header[f'PMIN{ii + 1}{letter}'] = ((a[param]["min"].to(unit).value - transb) / transa,
                                                           f'Minimum value of parameter {name} '
                                                           f'for component {ii + 1}')
                    hdu.header[f'PTRA{ii + 1}{letter}'] = (transa, 'Linear coefficient A in Lambda=PVAL*PTRA+PTRB')
                    hdu.header[f'PTRB{ii + 1}{letter}'] = (transb, 'Linear coefficient B in Lambda=PVAL*PTRA+PTRB')

                    hdu.header[f'PCONS{ii + 1}{letter}'] = (
                        0, f'1 if parameter {name}] for component {ii + 1} is constant')
                    if "constrain" in a[param]:
                        if a[param]["constrain"]["operation"] == "constant":
                            hdu.header[f'PCONS{ii + 1}{letter}'] = (
                                1, f'1 if parameter {name}] for component {ii + 1} is constant')
                        else:
                            raise NotImplementedError
                    last_index_ = f'PCONS{ii + 1}{letter}'

            elif self.components_results[key]["info"]["type"] == "polynomial":
                hdu.header[f'CMPTYP{str(ii + 1)}'] = ("Polynomial", f"Type of fit component {ii + 1}")
                hdu.header[f'CMPNAM{ii + 1}'] = (key, f'Name of fit component {ii + 1}')
                hdu.header[f'CMP_NP{ii + 1}'] = (len(subkeys), f'Number of parameters in fit component {ii + 1}')
                ncoef = len(a.keys())
                for jj, letter in enumerate(string.ascii_lowercase[:ncoef]):
                    data[index, ...] = self.components_results[key]["coeffs"][letter][results_type]
                    index = index + 1
                    transa = 1
                    transb = 0

                    Letter = string.ascii_uppercase[jj]
                    hdu.header[f'CMPDES{ii + 1}{Letter}'] = (f'This component is a polynomial of degree'
                                                             f' {len(a.keys()) - 1}',
                                                             )
                    hdu.header[f'CMPDES{ii + 1}{Letter}'] = (f'This component is a polynomial of degree'
                                                             f' {len(a.keys()) - 1}',
                                                             )
                    hdu.header[f'PNAME{ii + 1}{Letter}'] = (f'{letter}{ii + 1}',
                                                            f'Name of parameter {Letter} '
                                                            f'for component {ii + 1}'
                                                            )
                    hdu.header[f'PUNIT{ii + 1}{Letter}'] = (str(a[letter]['results'].unit),
                                                            f'Name of parameter {Letter} '
                                                            f'for component {ii + 1}'
                                                            )
                    hdu.header[f'PDESC{ii + 1}{Letter}'] = (f'This is the coefficient for x^{ii}',
                                                            f'Description of parameter {Letter} '
                                                            f'for component {ii + 1}'
                                                            )
                    unit = a[letter]['results'].unit

                    hdu.header[f'PINIT{ii + 1}{Letter}'] = (
                        (a[letter]["guess"].to(unit).value - transb) / transa,
                        f'Initial Value of parameter {Letter} '
                        f'for component {ii + 1}'
                    )
                    hdu.header[f'PMAX{ii + 1}{Letter}'] = ((a[letter]["max"].to(unit).value - transb) / transa,
                                                           f'Maximum Value of parameter {Letter} '
                                                           f'for component {ii + 1}'
                                                           )
                    hdu.header[f'PMIN{ii + 1}{Letter}'] = ((a[letter]["min"].to(unit).value - transb) / transa,
                                                           f'Minimum Value of parameter {Letter} '
                                                           f'for component {ii + 1}'
                                                           )
                    hdu.header[f'PTRA{ii + 1}{Letter}'] = (transa, 'Linear coefficient A in Lambda=PVAL*PTRA+PTRB'
                                                           )
                    hdu.header[f'PTRB{ii + 1}{Letter}'] = (transb, 'Linear coefficient B in Lambda=PVAL*PTRA+PTRB'
                                                           )
                    hdu.header[f'PCONS{ii + 1}{Letter}'] = (
                        0, f'1 if parameter {Letter}] for component {ii + 1} is constant')
                    if "constrain" in a[letter]:
                        if a[letter]["constrain"]["operation"] == "constant":
                            hdu.header[f'PCONS{ii + 1}{Letter}'] = (
                                1, f'1 if parameter {Letter}] for component {ii + 1} is constant')
                        else:
                            raise NotImplementedError
                    last_index_ = f'PCONS{ii + 1}{letter}'

            elif (self.components_results[key]["info"]["type"] == "chi2"):
                data[index, ...] = self.components_results[key]["coeffs"]["chi2"]["results"]
                index = index + 1
                hdu.header[f'CMPTYP{str(ii + 1)}'] = ("Polynomial"), f"Type of fit component {ii + 1}"
                hdu.header[f'CMPNAM{ii + 1}'] = ('Error of fit curve (Chi^2)', f'Name of component {ii + 1}')
                hdu.header[f'CMP_NP{ii + 1}'] = (len(subkeys), f'Number of parameters in component {ii + 1}')
                last_index_ = f'CMP_NP{ii + 1}'
            elif self.components_results[key]["info"]["type"] == "flagged_pixels":
                last_index_ = None
            if last_index_ is not None:
                last_index.append(last_index_)
        hdu.header.insert('FILENAME', ('', '-------------------------------------'), after=True)
        hdu.header.insert('FILENAME', ('', '| Keywords describing the whole ANA |'), after=True)
        hdu.header.insert('FILENAME', ('', '-------------------------------------'), after=True)
        hdu.header.insert('FILENAME', ('', '      '), after=True)
        hdu.header.insert('FILENAME', ('', '      '), after=True)
        hdu.header.insert('FILENAME', ('', '      '), after=True)
        for ii, last_index_ in enumerate(last_index[:-1]):
            hdu.header.insert(last_index_, ('', '-------------------------------------'), after=True)
            hdu.header.insert(last_index_, ('', f'| Keywords describing fit component {ii + 1} |'), after=True)
            hdu.header.insert(last_index_, ('', '-------------------------------------'), after=True)
            hdu.header.insert(last_index_, ('', '      '), after=True)
            hdu.header.insert(last_index_, ('', '      '), after=True)
            hdu.header.insert(last_index_, ('', '      '), after=True)
        hdu.header["BTYPE"] = '        '
        hdu.header["UCD"] = '        '
        hdu.header["BUNIT"] = '        '
        hdu.header["NWIN"] = (5, "Number of windows")
        hdu.header["WINNO"] = (0, "Number of windows")
        hdu.header["ANA_MISS"] = 'NaN     '
        hdu.header["WCSNAME"] = 'Helioprojective-cartesian'

        for jj in [1, 2]:
            hdu.header[f"CTYPE{jj}"] = header_ref[f"CTYPE{jj}"]
            hdu.header[f"CUNIT{jj}"] = header_ref[f"CUNIT{jj}"]
            hdu.header[f"CRVAL{jj}"] = header_ref[f"CRVAL{jj}"]
            hdu.header[f"CDELT{jj}"] = header_ref[f"CDELT{jj}"]
            hdu.header[f"CRPIX{jj}"] = header_ref[f"CRPIX{jj}"]

            hdu.header[f"CRDER{jj}"] = header_ref[f"CRDER{jj}"]
            hdu.header[f"CWERR{jj}"] = header_ref[f"CWERR{jj}"]
        hdu.header["PC1_1"] = header_ref[f"PC1_1"]
        hdu.header["PC1_2"] = header_ref[f"PC1_2"]
        hdu.header["PC2_2"] = header_ref[f"PC2_1"]
        hdu.header["PC2_2"] = header_ref[f"PC2_2"]
        if "CUNIT4" in header_ref:
            hdu.header[f"CTYPE3"] = header_ref[f"CTYPE4"]
            hdu.header[f"CUNIT3"] = header_ref[f"CUNIT4"]
            hdu.header[f"CRVAL3"] = header_ref[f"CRVAL4"]
            hdu.header[f"CDELT3"] = header_ref[f"CDELT4"]
            hdu.header[f"CRPIX3"] = header_ref[f"CRPIX4"]

            # hdu.header[f"CRDER4"] = header_ref[f"CRDER4"]
            # hdu.header[f"CWERR4"] = header_ref[f"CWERR4"]

            hdu.header[f"PC3_3"] = header_ref[f"PC4_4"]
            hdu.header[f"PC3_1"] = header_ref[f"PC4_1"]
        hdu.header["CTYPE4"] = 'FITCMP'
        hdu.header["CUNIT4"] = '        '
        hdu.header["CRVAL4"] = 1.00000
        hdu.header["CDELT4"] = 1.00000
        hdu.header["CRPIX4"] = 1.00000
        hdu.header["PC4_4"] = 1.00000
        key_list = [
            "SPECSYS", "VELOSYS",
            "DSUN_OBS", "DSUN_AU",
            "RSUN_ARC", "RSUN_REF",
            "SOLAR_B0", "SOLAR_P0", "SOLAR_EP",
            "CAR_ROT",
            "HGLT_OBS", "HGLN_OBS", "CRLT_OBS", "CRLN_OBS",
            "HEEX_OBS", "HEEY_OBS", "HEEZ_OBS",
            "HCIX_OBS", "HCIY_OBS", "HCIZ_OBS",
            "HCIX_VOB", "HCIY_VOB", "HCIZ_VOB",
            "HAEX_OBS", "HAEY_OBS", "HAEZ_OBS",
            "HEQX_OBS", "HEQY_OBS", "HEQZ_OBS",
            "GSEX_OBS", "GSEY_OBS", "GSEZ_OBS",
            "OBS_VR", "EAR_TDEL", "SUN_TIME",
            "DATE_EAR", "DATE_SUN",
            "XPOSURE",
            "TIMESYS",
            "DATEREF", "DATE-BEG", "DATE-OBS", "DATE-AVG", "DATE-END",
            "SEQ_BEG", "TELAPSE", "OBT_BEG",
            "INSTRUME", "OBSRVTRY", "CROTA"
        ]
        for k in key_list:
            hdu.header[k] = header_ref[k]
        hdu.header["LEVEL"] = 'L3      '
        hdu.header["CREATOR"] = 'Antoine Dolliou'
        hdu.header["ORIGIN"] = 'Max Planck Institute for Solar System Research'
        last_index_ = "BTYPE"
        hdu.header.insert(last_index_, ('', '      '))
        hdu.header.insert(last_index_, ('', '      '))
        hdu.header.insert(last_index_, ('', '      '))
        hdu.header.insert(last_index_, ('', '-------------------------------------'))
        hdu.header.insert(last_index_, ('', '| Keywords valid for this HDU |'))
        hdu.header.insert(last_index_, ('', '-------------------------------------'))

        return data

    def from_fits(self, path_to_fits: str = None, hdul=None,
                  verbose=0):
        """
        Creates a Fits_results object from a fits file, either through a
        path to a FITS object or a HDULIST object directly.
        The FITS file must have the save format given by the self.to_fits method
        :param verbose:
        :param path_to_fits: path to the FITS file.
        :param hdul: hdulist object of the FITS file.


        """

        if (path_to_fits is not None) and (hdul is None):
            hdul = fits.open(path_to_fits)
        elif (path_to_fits is None) and (hdul is not None):
            hdul = hdul
        else:
            raise ValueError("path_to_fits and hdul are both None or set to values. Only one of them must be set.")
        hdu0 = hdul[0]
        orname = hdu0.header["ORNAME"]

        hdu_parinfo = hdul[f"{orname} parinfo"]
        data_par = hdu_parinfo.data
        bytes = data_par['parinfo'][0]
        # sss = bytes.decode("UTF-8")
        parinfo = ast.literal_eval(bytes)
        fit_template = FittingModel(parinfo=parinfo)

        hdu_re = hdul[f"{orname} reconstruction"]
        data_re = hdu_re.data
        header_re = hdu_re.header
        ncoeff = header_re["NCOEFF"]
        coeffs_cp = data_re[:ncoeff, ...]
        coeffs_error_cp = data_re[ncoeff:(ncoeff + ncoeff), ...]
        fit_chit2_all = data_re[(ncoeff + ncoeff):(ncoeff + ncoeff + 1), ...]
        fit_chit2_all = fit_chit2_all.reshape(fit_chit2_all[0, ...].shape)
        flagged_pixels_ = data_re[(ncoeff + ncoeff + 1):(ncoeff + ncoeff + 2), ...]
        flagged_pixels = np.zeros_like(flagged_pixels_, dtype="bool")
        flagged_pixels[flagged_pixels_ > 0.5] = True
        flagged_pixels = flagged_pixels.reshape(flagged_pixels[0, ...].shape)
        self.verbose = verbose
        self.fit_template = fit_template
        self.fit_results = {"coeff": coeffs_cp,
                            "chi2": copy.deepcopy(fit_chit2_all),
                            "flagged_pixels": flagged_pixels,
                            "name": self.fit_template.params_free["notation"],
                            "unique_index": self.fit_template.params_free["unique_index"],
                            "coeff_name": self.fit_template.params_free["notation"],
                            "coeffs_error": coeffs_error_cp,
                            "unit": self.fit_template.params_free["unit"],
                            "trans_a": self.fit_template.params_free["trans_a"],
                            "trans_b": self.fit_template.params_free["trans_b"]}

        self.build_components_results()

        hdu_l2 = hdul[2]
        spectral_window = SpiceRasterWindowL2(hdu=hdu_l2)
        self.spectral_window = spectral_window
        if self.spectral_window.uncertainty is None:
            self.spectral_window.compute_uncertainty()

        return self

    # def gen_shmm(self, spicewindow: SpiceRasterWindowL2):
