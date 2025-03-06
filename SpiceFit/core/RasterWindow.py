from abc import ABC
from .FittingModel import FittingModel



class RasterWindowL2(ABC):
    def __init__(self):
        self.data = None
        self.header = None
        self.wcs = None

    def check_if_line_within(
        self, fittemplate: FittingModel, all_interval=False) -> bool:
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

    def return_wavelength_array(self):
        pass

    def return_wavelength_interval(self):
        pass

    def return_point_pixels(self, type="xyl"):
        pass

    def compute_uncertainty():
        pass

    def average_spectra_over_region():
        pass

    def return_lines_within(self):
        pass

    def return_fov(self):
        pass