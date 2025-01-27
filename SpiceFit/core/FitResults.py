import numpy as np
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
            fit_template: FittingModel,
            verbose=0,
    ):
        """

        :param fit_template: Fit template class
        :param verbose: Print text to console
        """

        self.verbose = verbose
        self.fit_template = fit_template
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

        self.fit_results = {
            "coeff": [],  # (N, time, Y , X)
            "chi2": [],  # (time, Y , X)
            "flagged_pixels": [],  # (time, Y , X)
            "radiance": [],  # (time, Y , X)
            "name": self.fit_template.params_free["notation"],
            "coeff_unit": []
        }

        self.lock = None
        self.cpu_count = None
        self.display_progress_bar = None
        self.spicewindow = None

        self.components_results = {}

    def fit_spice_window_standard(self,
                                  spicewindow: SpiceRasterWindowL2,
                                  parallelism: bool = True,
                                  cpu_count: int = None,
                                  min_data_points: int = 5,
                                  chi2_limit: float = 20.0, ):
        """

        Fit all pixels of the field of view for a given SpiceRasterWindowL2 class instance.

        :param spicewindow: SpiceRasterWindowL2 class
        :param parallelism: allow parallelism or not.
        :param cpu_count: cpu counts to use for parallelism.
        :param min_data_points: minimum data points for each pixel with for the fitting.
        :param chi2_limit: limit the chi^2 for a pixel. Above this value, the pixel will be flagged.
        :param display_progress_bar: display the progress bar
        """

        if spicewindow.uncertainty is None:
            spicewindow.compute_uncertainty()

        data_cube = u.Quantity(spicewindow.data, spicewindow.header["BUNIT"])
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

        self.spicewindow = spicewindow

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

        self.fit_results["index_in_template"] = self.fit_template.params_free["index"]
        self.fit_results["coeff_name"] = self.fit_template.params_free["name"]
        self.fit_results["coeff"] = coeffs_cp
        self.fit_results["coeffs_error"] = coeffs_error_cp
        self.fit_results["chi2"] = copy.deepcopy(fit_chit2_all)
        self.fit_results["flagged_pixels"] = copy.deepcopy(flagged_pixels)
        self.fit_results["unit"] = flatten(self.fit_template.params_free["unit"])

        self.build_components_results(self.components_results)

        shmm_data.close()
        shmm_data.unlink()
        shmm_lambda_.close()
        shmm_lambda_.unlink()
        shmm_uncertainty.close()
        shmm_uncertainty.unlink()
        shmm_fit_coeffs.close()
        shmm_fit_coeffs.unlink()
        shmm_fit_chi2.close()
        shmm_fit_chi2.unlink()
        shmm_flagged_pixels.close()
        shmm_flagged_pixels.unlink()

    def build_components_results(self, components_results: dict):
        type_list, index_list, coeff_list = self.fit_template.gen_mapping_params()
        for type_, index_, coeff_ in zip(type_list, index_list, coeff_list):
            a = self.fit_template.params_all[type_][index_][coeff_]
            wha = np.where(a["index"] == self.fit_results["index"])[0][0]
            if a["name_component"] not in self.components_results:
                self.components_results[a["name_component"]] = {"name_component": a["name_component"],}
            if a["free"]:
                self.components_results[a["name_component"]][coeff_] = self.fit_results["coeff"][wha, ...]
            else:
                dict_const = a["type_constrain"]
                b = self.fit_template.gen_coeff_from_unique_index(dict_const["ref"])
                whb = np.where(b["index"] == self.fit_results["index"])[0][0]
                if dict_const["operation"] == "plus":
                    self.components_results[a["name_component"]][coeff_] = self.fit_results["coeff"][whb, ...] + \
                                                                              dict_const["value"]
                elif dict_const["operation"] == "minus":
                    self.components_results[a["name_component"]][coeff_] = self.fit_results["coeff"][whb, ...] - \
                                                                              dict_const["value"]
                elif dict_const["operation"] == "times":
                    self.components_results[a["name_component"]][coeff_] = self.fit_results["coeff"][whb, ...] * \
                                                                              dict_const["value"]
                else:
                    raise NotImplementedError

            self.components_results[a["name_component"]][coeff_] = \
                u.Quantity(self.components_results[a["name_component"]][coeff_],self.fit_results["unit"][wha])
            self.components_results[a["name_component"]]["chi2"] = self.fit_results["chi2"][wha, ...]
        for type_, index_, coeff_ in zip(type_list, index_list, coeff_list):

            if (type_ == "gaussian") and ("radiance" not in self.components_results[a["name_component"]].keys()):
                a = self.fit_template.params_all[type_][index_][coeff_]
                wha = np.where(a["index"] == self.fit_results["index"])[0][0]

                I = self.components_results[a["name_component"]]["I"]
                x = self.components_results[a["name_component"]]["x"]
                s = self.components_results[a["name_component"]]["s"]

                line = None
                for line_ in self.fit_template.parinfo["info"]:
                    if line_["name"] == a["name_line"]:
                        line = line_
                if line is None:
                    raise NotImplementedError
                lambda_ref = u.Quantity(line["wave"], (line["unit_wave"]))
                self.components_results[a["name_component"]]["velocity"] = \
                    (const.c.to("km/s") * (x - lambda_ref)/lambda_ref).to(Constants.conventional_velocity_units)


                self.components_results[a["name_component"]]["radiance"] = \
                    (I * np.sqrt(2 * np.pi * s * s)).to(Constants.conventional_radiance_units)

                self.components_results[a["name_component"]]["fwhm"] = 2.355 * s


        for line_ in self.components_results.keys():
            if line_ ==self.fit_template.parinfo["main_line"]:
                self.components_results["main"] = self.components_results[a["name_component"]]




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
            breakpoint()
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

    # def _prepare_fitting_parameters(self):
    #     p0 = flatten(self.fit_template.parinfo["fit"]["guess"])
    #     max_arr = flatten(self.fit_template.parinfo["fit"]["max_arr"])
    #     min_arr = flatten(self.fit_template.parinfo["fit"]["min_arr"])
    #     unit = flatten(self.fit_template.parinfo["fit"]["units"])
    #     trans_a = flatten(self.fit_template.parinfo["fit"]["trans_a"])
    #     trans_b = flatten(self.fit_template.parinfo["fit"]["trans_b"])
    #
    #     self.fit_guess = []
    #     self.fit_max_arr = []
    #     self.fit_min_arr = []
    #     self._unit_coeffs_during_fitting = []
    #
    #     # convert into the right units ("W/ (m2 sr nm)" and "nm)
    #     for jj in range(len(p0)):
    #         p = u.Quantity(p0[jj], unit[jj]) * trans_a[jj] + u.Quantity(
    #             trans_b[jj], unit[jj]
    #         )
    #         mx = u.Quantity(max_arr[jj], unit[jj]) * trans_a[jj] + u.Quantity(
    #             trans_b[jj], unit[jj]
    #         )
    #         mn = u.Quantity(min_arr[jj], unit[jj]) * trans_a[jj] + u.Quantity(
    #             trans_b[jj], unit[jj]
    #         )
    #
    #         p = self._transform_to_conventional_unit(p)
    #         mx = self._transform_to_conventional_unit(mx)
    #         mn = self._transform_to_conventional_unit(mn)
    #
    #         self.fit_guess.append(p.value)
    #         self.fit_max_arr.append(mx.value)
    #         self.fit_min_arr.append(mn.value)
    #         if (p.unit != mx.unit) or (p.unit != mn.unit):
    #             raise ValueError("Not consistent units among fitting parameters")
    #         self._unit_coeffs_during_fitting.append(p.unict)

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
        if self.spicewindow is None:
            raise ValueError("The data is still not fitted")

        w_xy = self.spicewindow.w_xy

        # x, y = np.meshgrid(np.arange(self.spicewindow.data.shape[3]), np.arange(self.spicewindow.data.shape[2]))
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
        coeff[not self.fit_results["flagged_pixels"][0, :, :]] = np.nan
        im = ax.imshow(coeff, origin="lower", interpolation="none",
                       cmap=cmap)
        plt.colorbar(im, ax=ax)

        fig.savefig("test.pdf")

    def quicklook(self, line="main", show=True):
        """
        Function to quickly plot the main results of a fitted line
        :param line:
        """
        if line not in self.components_results.keys():
            raise ValueError(f"Cannot plot {line} as it is not a fitted line")
        a = self.components_results[line]
        cm = Constants.inch_to_cm
        cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
        cmap.set_bad('white')

        fig = plt.figure(figsize=(17*cm, 17*cm))
        gs = GridSpec(2, 2, wspace=0.7, hspace=0.7)
        axs = [fig.add_subplot(gs[i, j]) for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]) ]
        for ii, param, unit in zip(range(4), ["radiance", "velocity", "fwhm", "chi2"],
                                   ["W/ (m2 sr)", "km/s", "nm", None]):
            if unit is not None:
                im = axs[ii].imshow(a[param].to(unit).value, origin="lower", interpolation="none",cmap = cmap)
                cbar = fig.colorbar(im, ax=axs[ii], label=unit)

            else:
                im = axs[ii].imshow(a[param], origin="lower", interpolation="none", cmap=cmap)
                cbar = fig.colorbar(im, ax=axs[ii])
            axs[ii].set_title(param)
            if show == True:
                fig.show()
            else:
                return fig


    def organize_components(self):
        """
        create the self.component dictionnary, with all the fitting and derived parameters for the lines.
        The "main" entrance is the main line of the fit_template attribute
        The components format will depend on the fit type ("gaussian", "background")
        """
        self.components = {
            "chi2": self.fit_chi2,
            "flagged": self.fla
        }
        types = self.fit_template.parinfo["fit"]["type"]
        main_lines = self.fit_template.parinfo["main_line"]
        names = self.fit_template.parinfo["fit"]["names"]
        n_components = self.fit_template.parinfo["fit"]["n_components"]
        n = 0
        for ii, name in enumerate(names):
            type = types[ii]

            self.components[name] = {
                "fit_coeff": self.fit_results["coeff"][n:n + n_components, ...],
                "fit_coeff_unit": self.fit_results["coeff_unit"][n:n + n_components],
                "parinfo_fit": self.fit_template.get_component_fit(component_name=name),
            }
            if type == "gauss":
                self.components[name]["parinfo_info"] = self.fit_template.get_component_info(component_name=name)

            n = n + n_components

    def check_spectra(self, position="random"):
        """

        :param position:
        """
        pass

    # def gen_shmm(self, spicewindow: SpiceRasterWindowL2):
