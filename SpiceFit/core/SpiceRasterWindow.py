import numpy as np
import astropy.io.fits.header
from .FitTemplate import FitTemplate
from sospice.calibrate import spice_error
from astropy.wcs import WCS
import copy


class SpiceRasterWindowL2:

    def __init__(
        self,
        hdu: astropy.io.fits.hdu.image.ImageHDU = None,
        data: np.ndarray = None,
        header: astropy.io.fits.header.Header = None,
    ) -> None:
        """

        :param hdu:
        SPICE L2 FITS hdu
        :param data:
        SPICE L2 4D ndarray data (t, lambda, y, x). Assumed in W/m2/sr/nm
        :param header:
        SPICE L2 hdu header
        """
        if hdu is not None:

            self.data = copy.deepcopy(hdu.data)
            self.header = copy.deepcopy(hdu.header)
        elif (data is not None) and (header is not None):
            if hdu is None:
                self.data = copy.deepcopy(data)
                self.header = copy.deepcopy(header)
            else:
                raise RuntimeError(
                    "SpiceRasterwindow: Please choose between hdu or data and header to set to None."
                )
        else:
            raise RuntimeError("SpiceRasterWindow: incorrect input for initialization.")
        if self.header["LEVEL"] != "L2":
            raise RuntimeError("SpiceRasterWindow: FITS LEVEL not set to L2.")
        self.uncertainty = None
        self.uncertainty_average = None

        # Building the different wcs depending on the axes
        self.wcs = WCS(self.header)
        self.w_xyt = copy.deepcopy(self.wcs).dropaxis(2)
        self.w_xy = copy.deepcopy(self.w_xyt)
        self.w_xy.wcs.pc[2, 0] = 0
        self.w_xy = self.w_xy.dropaxis(2)
        self.w_spec = copy.deepcopy(self.wcs).sub(["spectral"])

    def check_if_line_within(
        self, fittemplate: FitTemplate, all_interval=False
    ) -> bool:
        """
        Check if a given line is within the window.
        :param fittemplate: fittemplate of the line to check
        :param all_interval: if True, then the entire interval guess of the line must be within the window. If no,
                             if False, then just the central wavelength must be within the window
        """
        wave_interval = self.return_wavelength_interval()
        if all_interval:
            if (fittemplate.wave_interval[0] > wave_interval[0]) and (
                fittemplate.wave_interval[1] < wave_interval[1]
            ):
                return True
            else:
                return False
        else:
            if (fittemplate.central_wave > wave_interval[0]) and (
                fittemplate.central_wave < wave_interval[1]
            ):
                return True
            else:
                return False

    def return_lines_within(self):
        pass

    def return_fov(self):
        pass

    def return_wavelength_interval(self) -> list:
        lenlambda = self.data.shape[1]
        z = np.arange(lenlambda)
        lamb = self.w_spec.world_to_pixel(z)
        dlamb = lamb[1] - lamb[0]
        return [lamb[0] - 0.5 * dlamb, lamb[-1] + 0.5 * dlamb]

    def compute_uncertainty(self, verbose: bool = False) -> None:
        av_constant_noise_level, sigma = spice_error(
            data=self.data, header=self.header, verbose=verbose
        )
        self.uncertainty = sigma
        self.uncertainty_average = av_constant_noise_level

    def return_point_pixels(self, type="xylt"):
        if type == "xylt":
            t, l, y, x = np.meshgrid(
                np.arange(self.data.shape[0]),
                np.arange(self.data.shape[1]),
                np.arange(self.data.shape[2]),
                np.arange(self.data.shape[3]),
                indexing="ij",
            )
            return x, y, l, t
        elif type == "xyl":
            l, y, x = np.meshgrid(
                np.arange(self.data.shape[1]),
                np.arange(self.data.shape[2]),
                np.arange(self.data.shape[3]),
                indexing="ij",
            )
            return x, y, l
        elif type == "xy":
            y, x = np.meshgrid(
                np.arange(self.data.shape[2]),
                np.arange(self.data.shape[3]),
                indexing="ij",
            )
            return x, y
        elif type == "l":
            l = np.meshgrid(np.arange(self.data.shape[1]), indexing="ij")
            return l
