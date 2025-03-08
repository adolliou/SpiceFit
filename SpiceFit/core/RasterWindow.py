from abc import ABC
from .FittingModel import FittingModel
from astropy.coordinates import SkyCoord
import numpy as np
from ..util.plotting_fits import PlotFits
import astropy.units as u


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

    def compute_uncertainty(self,):
        pass

    def return_lines_within(self):
        pass

    def return_fov(self):
        pass

    def extract_subfield_coordinates(self,coords: SkyCoord = None, lonlat_lims: tuple = None,
                                     pixels: tuple = None, allow_reprojection: bool = False,):
        """
            extract coordinates in pixels and in world coordinates of a given subfov.
            called by average_spectra_over_region. Please refer to this function
            to see description of the parameters
        Args:
            coords (SkyCoord, optional): skycoord coordinate object where to average the spectra. Defaults to None.
            lonlat_lims (tuple, optional): limits in longitude and latitude where to average the spectra.
            Format is ((240*u.arcsec, 260*u.arcsec,), (-25*u.arcsec, 25*u.arcsec,)). Defaults to None.
            pixels (tuple, optional): pixels where to average the spectra. format is (x, y). Defaults to None.
            seld.data[:, : y, x]. x refers to the longitude.
            allow_reprojection(bool, optional) if set to true, then the spectrum can be reprojected over the given region.
            The spatial reprojection (the both the data and the sigma) is done for each wavelength step individually.
            If false, the the spectrum is directly taken from the given pixels (or their given coordinates)
        """
        error_str_lonlatlims = "lonlat_lims has the wrong format. it should be of the form ((240*u.arcsec, 260*u.arcsec,), (-25*u.arcsec, 25*u.arcsec,))"
        error_str_coords = "coords has the wrong format. it should be a astropy.coordintes.SkyCoord object"
        error_str_pixels = "pixels has the wrong format. it should be a tuple (x, y)"
        if coords is not None:
            try:
                if not ('Tx' in coords.representation_component_names):
                    raise ValueError("Frame in Skycoord not recognised. Can only use helioprojective frame as of now")
                lon = coords.Tx
                lat = coords.Ty
                x, y = self.w_xy.world_to_pixel(coords)
                x = x.ravel()
                y = y.ravel()
            except:
                raise ValueError(error_str_coords)
        if lonlat_lims is not None:
            if not (allow_reprojection):
                raise ValueError(
                    "Please allow reprojection if you want to average over a given region in longitude/latitudes limits.")
            if isinstance(lonlat_lims, tuple) & (len(lonlat_lims) == 2) & isinstance(lonlat_lims[0], tuple) & \
                    isinstance(lonlat_lims[0], tuple) & isinstance(lonlat_lims[0][0], u.Quantity):

                x, y = self.return_point_pixels(type="xy")
                coords_tmp = self.w_xy.pixel_to_world(x, y)
                lon_tmp = coords_tmp.Tx
                lat_tmp = coords_tmp.Ty
                lonlim = lonlat_lims[0]
                latlim = lonlat_lims[1]
                lon, lat, dlon, dlat = PlotFits.build_regular_grid(lon_tmp, lat_tmp, lonlims=lonlim, latlims=latlim, )

                dlon = dlon.to(lon.unit).value
                dlat = dlat.to(lat.unit).value

                coords = SkyCoord(lon, lat, frame=coords_tmp.frame)
                lon = coords.Tx
                lat = coords.Ty
                x, y = self.w_xy.world_to_pixel(coords)
                x = x.ravel()
                y = y.ravel()

            else:
                raise ValueError(error_str_lonlatlims)
        if pixels is not None:
            try:
                if (len(pixels) == 2) & isinstance(pixels, tuple):
                    x = np.array(pixels[0]).ravel()
                    y = np.array(pixels[1]).ravel()
                    assert len(x) == len(y)
                    coords = self.w_xy.pixel_to_world(x, y)
                    lon = coords.Tx
                    lat = coords.Ty
                else:
                    ValueError
            except:
                raise ValueError(error_str_pixels)
        return lat, lon, x, y


    def average_spectra_over_region(self, coords: SkyCoord = None, lonlat_lims: tuple = None, pixels: tuple = None,
                                    allow_reprojection=False):
        """Average the spectra and error over a given spatial region.
        Returns a SpiceRasterWindow object.

        Args:
            coords (SkyCoord, optional): skycoord coordinate object where to average the spectra. Defaults to None.
            lonlat_lims (tuple, optional): limits in longitude and latitude where to average the spectra.
            Format is ((240*u.arcsec, 260*u.arcsec,), (-25*u.arcsec, 25*u.arcsec,)). Defaults to None.
            pixels (tuple, optional): pixels where to average the spectra. format is (x, y). Defaults to None.
            seld.data[:, : y, x]. x refers to the longitude.
            allow_reprojection(bool, optional) if set to true, then the spectrum can be reprojected over the given region.
            The spatial reprojection (the both the data and the sigma) is done for each wavelength step individually.
            If false, the spectrum is directly taken from the given pixels (or their given coordinates)
        """

        pass