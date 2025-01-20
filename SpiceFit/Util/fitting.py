import copy
import numpy as np
from matplotlib import pyplot as plt
import itertools



class FittingUtil:

    @staticmethod
    def gaussian(x: np.array, I: np.float64, mu: np.float64, sigma: np.float64, back: np.float64):
        return I * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + back



    @staticmethod
    def multiple_gaussian_cfit(x: np.array, *params):
        y = np.zeros_like(x, dtype=float)
        y += params[-1]
        if (len(params) - 1) % 3 != 0:
            raise ValueError("multiple_gaussian: inconsistent number of arguments")
        for ii in range(0, len(params)-1, 3):
            I = params[ii]
            mu = params[ii+1]
            sigma = params[ii+2]
            y += I * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        return y



    @staticmethod
    def multiple_gaussian(x: np.array, I: list, mu: list, sigma: list, back: np.float64):
        s = np.array([I_ * np.exp(-((x - mu_) ** 2) / (2 * sigma_ ** 2)) for I_, mu_, sigma_ in zip(I, mu, sigma)],
                     dtype=np.float64)
        return s.sum(axis=0) + back

    @staticmethod
    def polynomial(x, *coeff):
        """

        :param x:
        :param coeff: coeff = [a, b, c ] = cx**2 + bx + a = y
        :return: y
        """
        order = len(coeff)
        return np.array([coeff[i] * x ** i for i in range(order)]).sum(axis=0)


class PlotSpectrum:

    @staticmethod
    def _get_edges(x: np.array):

        edges = [x[n] + (x[n + 1] - x[n]) / 2 for n in range(len(x) - 1)]
        edges.append(x[-1] + (x[-1] - x[-2]) / 2)
        edges.insert(0, x[0] - (x[1] - x[0]) / 2)
        return edges

    @staticmethod
    def plot_spectrum(lam: np.array, spectrum: np.array, error_spectrum=None, label="rd", color="r",
                      linewidth_stair=0.5, linewidth_fit=0.5,
                      fig=None, ax=None, show_legend=True, save_path=None, fitting_function=None, kwargs_fitting=None,
                      kwargs_sigma_fitting=None, sigma=1
                      ):

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        sort = lam.argsort()

        lam = lam[sort]
        spectrum = spectrum[sort]
        isnan = np.isnan(spectrum)
        spectrum = spectrum[~isnan]
        lam = lam[~isnan]
        if error_spectrum is not None:
            error_spectrum = error_spectrum[~isnan]

        edges_lam = PlotSpectrum._get_edges(lam)
        ax.stairs(spectrum, edges=edges_lam, label=label, color=color, linewidth=linewidth_stair, )
        if error_spectrum is not None:
            ax.errorbar(x=lam, y=spectrum, yerr=0.5*error_spectrum, linestyle="", marker="", capsize=0,
                        elinewidth=linewidth_stair*0.5, color=color)
        fits_sigma = None
        fit = None
        if fitting_function is not None:

            if fitting_function == "gaussian":
                fit = FittingUtil.gaussian(lam, **kwargs_fitting)
                if (kwargs_sigma_fitting is not None) & (kwargs_fitting is not None):
                    kwargs_list = PlotSpectrum._extract_all_possible_fitting_kwargs(kwargs_fitting,
                                                                                    kwargs_sigma_fitting,
                                                                                    sigma)
                    fits_sigma = np.empty((len(kwargs_list), len(lam)), dtype=np.float64)
                    for ii, kwarg_tmp in enumerate(kwargs_list):
                        fits_sigma[ii, :] = FittingUtil.gaussian(lam, **kwarg_tmp)

            elif fitting_function == "multiple_gaussian":
                fit = FittingUtil.multiple_gaussian(lam, **kwargs_fitting)
                size_index = len(kwargs_fitting["I"])
                if (kwargs_sigma_fitting is not None) & (kwargs_fitting is not None):
                    kwargs_fitting_extended = PlotSpectrum._extend_kwarg(kwargs_fitting,
                                                                         single_key="back")
                    kwargs_fitting_sigma_extended = PlotSpectrum._extend_kwarg(kwargs_sigma_fitting,
                                                                               single_key="back")

                    kwargs_list = PlotSpectrum._extract_all_possible_fitting_kwargs(kwargs_fitting_extended,
                                                                                    kwargs_fitting_sigma_extended,
                                                                                    sigma)
                    fits_sigma = np.empty((len(kwargs_list), len(lam)), dtype=np.float64)
                    for ii, kwarg_tmp in enumerate(kwargs_list):
                        kwarg_tmp_reduced = PlotSpectrum._extend_kwarg(kwarg_tmp, inverse=True, size_index=size_index,
                                                                       single_key="back",
                                                                       keys=["I", "mu", "sigma", "back"])
                        fits_sigma[ii, :] = FittingUtil.multiple_gaussian(lam, **kwarg_tmp_reduced)

            ax.plot(lam, fit, color=color, linewidth=linewidth_fit, label="_nolegend_")
        if show_legend:
            ax.legend()
        if save_path is not None:
            fig.savefig(save_path)
        return edges_lam, spectrum, lam, fit, fits_sigma

    @staticmethod
    def _extend_kwarg(kwargs_fitting, inverse=False, keys=None, single_key=None, size_index=2):
        if inverse:
            if keys is None:
                keys = kwargs_fitting.keys()
            kwargs_fitting_reduced = {}
            for key in keys:
                if single_key != key:
                    kwargs_fitting_reduced[key] = []
                    for ii in range(size_index):
                        kwargs_fitting_reduced[key].append(copy.deepcopy(kwargs_fitting[f"{key}_{ii}"]))
                else:
                    kwargs_fitting_reduced[key] = copy.deepcopy(kwargs_fitting[f"{key}"])

            return kwargs_fitting_reduced
        else:
            if keys is None:
                keys = kwargs_fitting.keys()
            kwargs_fitting_extended = {}
            for key in keys:
                if single_key != key:
                    for ii in range(len(kwargs_fitting[key])):
                        kwargs_fitting_extended[f"{key}_{ii}"] = copy.deepcopy(kwargs_fitting[key][ii])
                else:
                    kwargs_fitting_extended[f"{key}"] = copy.deepcopy(kwargs_fitting[key])
            return kwargs_fitting_extended

    @staticmethod
    def _extract_all_possible_fitting_kwargs(kwargs_fitting, kwargs_sigma_fitting, sigma):
        list_kwarg_sigma = []
        keys = kwargs_sigma_fitting.keys()
        keys_plus_minus = []
        for key in keys:
            keys_plus_minus.append(f"{key}_plus")
            keys_plus_minus.append(f"{key}_minus")
        for ii in range(len(keys_plus_minus)):
            for subset in itertools.combinations(keys_plus_minus, ii):
                kwargs_fit_tmp = copy.deepcopy(kwargs_fitting)
                kwargs_fit_cp = copy.deepcopy(kwargs_fitting)
                for key in subset:
                    if "plus" in key:
                        key_or = key.replace("_plus", "")
                        kwargs_fit_tmp[key_or] = kwargs_fit_cp[key_or] + 0.5 * sigma * kwargs_sigma_fitting[key_or]
                    elif "minus" in key:
                        key_or = key.replace("_minus", "")
                        kwargs_fit_tmp[key_or] = kwargs_fit_cp[key_or] - 0.5 * sigma * kwargs_sigma_fitting[key_or]
                list_kwarg_sigma.append(copy.deepcopy(kwargs_fit_tmp))
        return list_kwarg_sigma
