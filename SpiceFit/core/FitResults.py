from FitTemplates import FitTemplate
from .SpiceRasterWindow import SpiceRasterWindowL2
from ..Util.shared_memory import gen_shmm
import copy


class FitResults:

    def __init__(self, fit_template: FitTemplate):
        """

        :param fit_template: Fit template class
        """
        self.fit_template = fit_template

        self.fitfunction = None

        self.data = None
        self.uncertainty = None
        self.weight = None
        self.fit_results = None
        self.chi2 = None

        self._data_dict = None
        self._uncertainty_dict = None
        self._weight_dict = None
        self._fit_results_dict = None
        self._chi2_dict = None

        self._shmm_data = None
        self._shmm_uncertainty = None
        self._shmm_weight = None
        self._shmm_fit_results = None
        self._shmm_chi2 = None


    def fit_window_standard(self, spicewindow: SpiceRasterWindowL2, parallelism: bool = True, ):
        """

        :param S: SpiceRasterWindowL2 class
        :param parallelism: allow parallelism or not.
        """

        if spicewindow.uncertainty is None:
            spicewindow.compute_uncertainty(verbose=False)
        if parallelism:
            self.gen_shmm(spicewindow)

    def gen_shmm(self, spicewindow: SpiceRasterWindowL2):
        self._shmm_data, data = gen_shmm(create=True, ndarray=copy.deepcopy(spicewindow.data))
        self._shmm_uncertainty, uncertainty = gen_shmm(create=True, ndarray=copy.deepcopy(spicewindow.uncertainty["Total"]))
        self._shmm_data, data = gen_shmm(create=True, ndarray=copy.deepcopy(spicewindow.data))
        self._shmm_data, data = gen_shmm(create=True, ndarray=copy.deepcopy(spicewindow.data))



        self._data_dict = {"name": self._shmm_data.name, "dtype": self._shmm_data.dtype, "shape": self._shmm_data.shape}

        self._shmm_data, data = gen_shmm(create=True, ndarray=copy.deepcopy(spicewindow.data))







