import numpy as np

from FitTemplates import FitTemplate
from .SpiceRasterWindow import SpiceRasterWindowL2
from ..Util.shared_memory import gen_shmm
from ..Util.fitting import FittingUtil
import copy
from multiprocessing import Process, Lock
import multiprocessing as mp


class FitResults:

    def __init__(self, fit_template: FitTemplate, verbose=False,):
        """

        :param fit_template: Fit template class
        :param verbose: Print text to console
        :param count: cpu counts to use for parallelism
        """

        self.verbose = verbose
        self.fit_template = fit_template
        self.fit_function_params = None

        self.fit_function = None

        self.data = None
        self.uncertainty = None

        self.fit_coeffs = None
        self.fit_chi2 = None

        self._data_dict = None
        self._uncertainty_dict = None
        self._weight_dict = None
        self._fit_coeffs_dict = None
        self._fit_chi2_dict = None

        # self._shmm_data = None
        # self._shmm_uncertainty = None
        # self._shmm_fit_coeffs = None
        # self._shmm_fit_chi2 = None

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
    ):
        """
        Fit all pixels of the field of view for a given SpiceRasterWindowL2 class instance.

        :param spicewindow: SpiceRasterWindowL2 class
        :param parallelism: allow parallelism or not.
        :param count: cpu counts to use for parallelism

        """
        if count is None:
            self.count = mp.cpu_count()
        else:
            self.count = count
        self._gen_function()

        if spicewindow.uncertainty is None:
            spicewindow.compute_uncertainty(verbose=False)
        if parallelism:
            self.lock = Lock()
            self.gen_shmm(spicewindow)

            # Get indexes and positions in (i, j) for all pixels of the raster
            i_array, j_array = np.meshgrid(np.arange(self.data.shape[0]), np.arange(self.data.shape[1]), indexing='ij')
            i_array = i_array.flatten()
            j_array = j_array.flatten()
            indexes = i_array*self.data.shape[1] + j_array

            processes = []

            for i, j, index in zip(i_array, j_array, indexes):
                kwargs = {
                    "i" : i,
                    "j" : j,
                    "index" : index,
                    "lock" : self.lock
                }

                processes.append(Process(target=self._fit_pixel_parallelism, kwargs=kwargs))

            lenp = len(processes)
            ii = -1
            is_close = []
            while ii < lenp - 1:
                ii += 1
                processes[ii].start()
                while (np.sum([p.is_alive() for mm, p in zip(range(lenp), processes) if
                               (mm not in is_close)]) > self.count):
                    pass
                for kk, P in zip(range(lenp), processes):
                    if kk not in is_close:
                        if (not (P.is_alive())) and (kk <= ii):
                            P.close()
                            is_close.append(kk)

            while np.sum([p.is_alive() for mm, p in zip(range(lenp), processes) if (mm not in is_close)]) != 0:
                pass
            for kk, P in zip(range(lenp), processes):
                if kk not in is_close:
                    if (not (P.is_alive())) and (kk <= ii):
                        P.close()
                        is_close.append(kk)

            shmm_data, data = gen_shmm(
                create=False, **self._data_dict
            )
            shmm_uncertainty, uncertainty = gen_shmm(
                create=False, **self._uncertainty_dict
            )
            shmm_fit_coeffs, fit_coeffs = gen_shmm(
                create=False, **self._fit_coeffs_dict
            )
            shmm_fit_chi2, fit_chit2 = gen_shmm(
                create=False, **self._fit_chi2_dict
            )

            # TODO save all results in permanent addresses before unlinking all shmm objects

            shmm_data.close()
            shmm_uncertainty.close()
            shmm_fit_coeffs.close()
            shmm_fit_chi2.close()

    def _fit_pixel_parallelism(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def gen_shmm(self, spicewindow: SpiceRasterWindowL2):
        shmm_data, data = gen_shmm(
            create=True, ndarray=copy.deepcopy(spicewindow.data)
        )
        shmm_uncertainty, uncertainty = gen_shmm(
            create=True, ndarray=copy.deepcopy(spicewindow.uncertainty["Total"])
        )
        shmm_fit_coeffs, fit_coeffs = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    spicewindow.data.shape[0],
                    spicewindow.data.shape[1],
                    np.sum(self.fit_template.parinfo["fit"]["n_components"]),
                ),
                dtype="float",
            ),
        )
        shmm_fit_chi2, fit_chi2 = gen_shmm(
            create=True,
            ndarray=np.zeros(
                (
                    spicewindow.data.shape[0],
                    spicewindow.data.shape[1],
                ),
                dtype="float",
            ),
        )

        self._data_dict = {
            "name": shmm_data.name,
            "dtype": data.dtype,
            "shape": data.shape,
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

        shmm_data.close()
        shmm_uncertainty.close()
        shmm_fit_coeffs.close()
        shmm_fit_chi2.close()
