from .FittingModel import FittingModel
from scipy.optimize import curve_fit
import numpy as np

#
# def fit_pixel(x: np.array, y: np.array, fitting_model: FittingModel, minimum_real_values=5, verbose=0):
#     """
#
#     :param x: lambda data
#     :param y: spectra data
#     :param fitting_model:
#     :param minimum_real_values:
#     :param verbose:
#     """
#     # check the nans
#
#     popt, pcov = curve_fit(
#         f=fitting_function,
#         xdata=x,
#         ydata=data,
#         p0=guess,
#         sigma=uncertainty,
#         bounds=(min_arr, max_arr),
#         nan_policy="raise",
#         method="trf",
#         full_output=False,
#         absolute_sigma=True,
#     )