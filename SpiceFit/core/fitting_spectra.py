from .FittingModel import FittingModel
from scipy.optimize import curve_fit
import numpy as np


def fit_spectra(x, y, dy, fit_template: FittingModel, minimum_data_points: int = 5, verbose: bool = False):
    # check which datapoints are nans
    isnotnan = np.logical_not(np.isnan(y))

    x = x[isnotnan]
    y = y[isnotnan]
    dy = dy[isnotnan]

    if isnotnan.sum() > minimum_data_points:

        try:
            popt, pcov = curve_fit(
                f=fit_template.fitting_function,
                xdata=x,
                ydata=y,
                sigma=dy,
                p0=fit_template.params_free["guess"],
                bounds=fit_template.params_free["bounds"],
                nan_policy="raise",
                method="trf",
                full_output=False,
                # absolute_sigma=True,
                jac=fit_template.jacobian_function,
            )

        except RuntimeError:
            if verbose == 2:
                print("Fitting failed")

            return None, None

        return popt, pcov
