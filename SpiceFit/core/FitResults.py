import numpy as np
from FitTemplates import FitTemplate
from .SpiceRasterWindow import SpiceRasterWindowL2
from ..Util.shared_memory import gen_shmm
from ..Util.fitting import FittingUtil
import copy
from multiprocessing import Process, Lock
import multiprocessing as mp
from scipy.optimize import curve_fit
import astropy.units as u


def flatten(xss):
    """
    https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        :param xss:
        :return:
    """
    return [x for xs in xss for x in xs]


class FitResults:
    conventional_lambda_units = "nm"
    conventional_spectral_units = "W/ (m2 sr nm)"

    def __init__(
        self,
        fit_template: FitTemplate,
        verbose=False,
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

        # fitting results
        self.fit_coeffs = None
        self.fit_chi2 = None

        # fitting parameters
        self.fit_guess = None  # size : (N)
        self.fit_max_arr = None  # size : (N)
        self.fit_min_arr = None  # size : (N)
        self.fit_unit = None  # size : (N)
        self.min_data_points = None

        # dicts to create and gain access to shared memory objects
        self._data_dict = None  # size : (time, lambda, Y , X)
        self._lambda_dict = None  # size : (time, lambda, Y , X)
        self._uncertainty_dict = None  # size : (time, lambda, Y , X)
        self._fit_coeffs_dict = None  # size : (N, time, Y , X)
        self._fit_chi2_dict = None  # size : (N, time, Y , X)
        self._flagged_pixels_dict = None  # size : (time, Y , X)

        self.fit_results = {}
        for jj, name in enumerate(self.fit_template.parinfo["fit"]["name"]):
            self.fit_results[name] = {
                "coeffs": [],
                "type": self.fit_template.parinfo["fit"]["type"],
                "max_arr": self.fit_template.parinfo["fit"]["max_arr"][jj],
                "min_arr": self.fit_template.parinfo["fit"]["max_arr"][jj],
                "trans_a": self.fit_template.parinfo["fit"]["max_arr"][jj],
                "trans_b": self.fit_template.parinfo["fit"]["max_arr"][jj],
                "n_components": self.fit_template.parinfo["fit"]["ncomponents"][jj],
            }

        self.lock = None
        self.count = None

    def fit_window_standard(
        self,
        spicewindow: SpiceRasterWindowL2,
        parallelism: bool = True,
        count: int = None,
        min_data_points: int = 5,
    ):
        """
        Fit all pixels of the field of view for a given SpiceRasterWindowL2 class instance.

        :param spicewindow: SpiceRasterWindowL2 class
        :param parallelism: allow parallelism or not.
        :param count: cpu counts to use for parallelism.
        :param min_data_points: minimum data points for each pixel with for the fitting.

        """

        self.min_data_points = min_data_points
        if count is None:
            self.count = mp.cpu_count()
        else:
            self.count = count
        self._gen_function()

        self._prepare_fitting_parameters()

        if spicewindow.uncertainty is None:
            spicewindow.compute_uncertainty(verbose=False)
        if parallelism:
            self.lock = Lock()
            self.gen_shmm(spicewindow)

            # Get indexes and positions in (i, j) for all pixels of the raster
            i_array, j_array = np.meshgrid(
                np.arange(self.data.shape[0]),
                np.arange(self.data.shape[1]),
                indexing="ij",
            )
            i_array = i_array.flatten()
            j_array = j_array.flatten()
            indexes = i_array * self.data.shape[1] + j_array

            processes = []

            for i, j, index in zip(i_array, j_array, indexes):
                kwargs = {"i": i, "j": j, "index": index, "lock": self.lock}

                processes.append(
                    Process(target=self._fit_pixel_parallelism, kwargs=kwargs)
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
                    > self.count
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

            shmm_data, data = gen_shmm(create=False, **self._data_dict)
            shmm_lambda_, lambda_ = gen_shmm(create=False, **self._lambda_dict)

            shmm_uncertainty, uncertainty = gen_shmm(
                create=False, **self._uncertainty_dict
            )
            shmm_fit_coeffs, fit_coeffs = gen_shmm(
                create=False, **self._fit_coeffs_dict
            )
            shmm_fit_chi2, fit_chit2 = gen_shmm(create=False, **self._fit_chi2_dict)

            # TODO save all results in permanent addresses before unlinking all shmm objects

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

    def _fit_pixel_parallelism(self, i, j, index, lock):
        shmm_data_all, data_all = gen_shmm(create=False, **self._data_dict)
        shmm_lambda_all, lambda_all = gen_shmm(create=False, **self._lambda_dict)
        shmm_uncertainty_all, uncertainty_all = gen_shmm(
            create=False, **self._uncertainty_dict
        )
        shmm_fit_coeffs_all, fit_coeffs_all = gen_shmm(
            create=False, **self._fit_coeffs_dict
        )
        shmm_fit_chi2_all, fit_chit2_all = gen_shmm(create=False, **self._fit_chi2_dict)

        data = data_all[i, j]
        uncertainty = uncertainty_all[i, j]
        lambda_ = lambda_all[i, j]
        guess = self.fit_guess
        max_arr = self.fit_max_arr
        min_arr = self.fit_min_arr

    def _gen_function(self):
        names = self.fit_template.parinfo["fit"]["names"]
        only_gaussi_const = np.logical_and(
            not any([n in ["gauss", "const"] for n in names]),
            len(np.array(names)[names == "const"]) == 1,
        )
        if only_gaussi_const:
            self.fit_function = self._gen_function_only_gaussians()
        else:
            self.fit_function = self._gen_function_exec()

    def _gen_function_only_gaussians(self):
        names = self.fit_template.parinfo["fit"]["names"]

        # Set the constant at the -1 index
        index_background = np.where(np.array(names) == "const")[0]
        if index_background != np.array(names) - 1:
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

    def _prepare_fitting_parameters(self):
        p0 = flatten(self.fit_template.parinfo["fit"]["guess"])
        max_arr = flatten(self.fit_template.parinfo["fit"]["max_arr"])
        min_arr = flatten(self.fit_template.parinfo["fit"]["max_arr"])
        unit = flatten(self.fit_template.parinfo["fit"]["unit"])
        trans_a = flatten(self.fit_template.parinfo["fit"]["trans_a"])
        trans_b = flatten(self.fit_template.parinfo["fit"]["trans_b"])

        self.fit_guess = []
        self.fit_max_arr = []
        self.fit_min_arr = []
        self.fit_unit = []

        # convert into the right units ("W/ (m2 sr nm)" and "nm)
        for jj in range(len(p0)):
            p = u.Quantity(p0[jj], unit[jj]) * trans_a[jj] + u.Quantity(
                trans_b[jj], unit[jj]
            )
            mx = u.Quantity(max_arr[jj], unit[jj]) * trans_a[jj] + u.Quantity(
                trans_b[jj], unit[jj]
            )
            mn = u.Quantity(min_arr[jj], unit[jj]) * trans_a[jj] + u.Quantity(
                trans_b[jj], unit[jj]
            )

            p = self._transform_to_conventional_unit(p)
            mx = self._transform_to_conventional_unit(mx)
            mn = self._transform_to_conventional_unit(mn)

            self.fit_guess.append(p.value)
            self.fit_max_arr.append(mx.value)
            self.fit_min_arr.append(mn.value)

            if (p.unit != mx.unit) or (p.unit != mn.unit):
                raise ValueError("Not consistent units among fitting parameters")
            self.fit_unit.append(p.unit)

    @staticmethod
    def _transform_to_conventional_unit(q: u.Quantity) -> u.Quantity:
        """
        transform an u.quantity into either a nm or a W/ (m2 sr nm), which are the conventional units for the fitting.
        :param q:
        """
        if q.is_equivalent(FitResults.conventional_lambda_units):
            return q.to(FitResults.conventional_lambda_units)
        elif q.is_equivalent(FitResults.conventional_spectral_units):
            return q.to(FitResults.conventional_spectral_units)
        else:
            raise ValueError(f"Cannot convert {q} to conventional unit")

    def gen_shmm(self, spicewindow: SpiceRasterWindowL2):
        data_ = copy.deepcopy(spicewindow.data)
        data_ = u.Quantity(data_, spicewindow.header["BUNIT"])

        xx, yy, ll, tt = spicewindow.return_point_pixels()
        coords, lambda_, t = spicewindow.wcs.pixel_to_world(xx, yy, ll, tt)
        lambda_ = lambda_.to(FitResults.conventional_lambda_units).value
        data_ = data_.to(FitResults.conventional_spectral_units).value

        self.lambda_unit = FitResults.conventional_lambda_units
        self.data_unit = FitResults.conventional_spectral_units

        shmm_data, data = gen_shmm(create=True, ndarray=data_)
        shmm_lambda_, lambda_ = gen_shmm(create=True, ndarray=lambda_)
        shmm_uncertainty, uncertainty = gen_shmm(
            create=True,
            ndarray=copy.deepcopy(
                spicewindow.uncertainty["Total"]
                .to(FitResults.conventional_spectral_units)
                .value
            ),
        )
        shmm_fit_coeffs, fit_coeffs = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    np.sum(self.fit_template.parinfo["fit"]["n_components"]),
                    spicewindow.data.shape[0],
                    spicewindow.data.shape[2],
                    spicewindow.data.shape[3],
                ),
                dtype="float",
            ),
        )
        shmm_fit_chi2, fit_chi2 = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    spicewindow.data.shape[0],
                    spicewindow.data.shape[2],
                    spicewindow.data.shape[3],
                ),
                dtype="float",
            ),
        )

        shmm_flagged_pixels, flagged_pixels = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    spicewindow.data.shape[0],
                    spicewindow.data.shape[2],
                    spicewindow.data.shape[3],
                ),
                dtype="int16",
            ),
        )
        self._flagged_pixels_dict = None  # size : (time, Y , X)

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

        shmm_data.close()
        shmm_uncertainty.close()
        shmm_lambda_.close()
        shmm_fit_coeffs.close()
        shmm_fit_chi2.close()
        shmm_flagged_pixels.close()
