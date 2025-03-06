import numpy as np
import astropy.io.fits.header
from .FittingModel import FittingModel
from sospice.calibrate import spice_error
from astropy.wcs import WCS
import copy
import astropy.units as u
from ..util.spice_util import SpiceUtil
from astropy.coordinates import SkyCoord
from .RasterWindow import RasterWindowL2
import sunpy.map
import math
from ..util.common_util import CommonUtil



def rss(a: np.array, axis=0):
    return np.sqrt(np.nansum(np.power(a, 2), axis=axis))


class SpiceRasterWindowL2(RasterWindowL2):

    def __init__(
        self,
        hdu: astropy.io.fits.hdu.image.ImageHDU = None,
        data: np.ndarray = None,
        header: astropy.io.fits.header.Header = None,
        remove_dumbbells = True,
    ) -> None:
        """

        :param hdu:
        SPICE L2 FITS hdu
        :param data:
        SPICE L2 4D ndarray data (t, lambda, y, x). Assumed in W/m2/sr/nm
        :param header:
        SPICE L2 hdu header
        """
        super().__init__()

        self.remove_dumbbells = remove_dumbbells
        if hdu is not None:

            self.data = copy.deepcopy(u.Quantity(hdu.data, hdu.header["BUNIT"]))
            self.header = copy.deepcopy(hdu.header)
        elif (data is not None) and (header is not None):
            if hdu is None:
                self.data = copy.deepcopy(u.Quantity(data, header["BUNIT"]))
                self.header = copy.deepcopy(header)
            else:
                raise RuntimeError(
                    "SpiceRasterwindow: Please choose between hdu or data and header to set to None."
                )
        else:
            raise RuntimeError("SpiceRasterWindow: incorrect input for initialization.")
        if self.header["LEVEL"] != "L2":
            raise RuntimeError("SpiceRasterWindow: FITS LEVEL not set to L2.")
        if self.remove_dumbbells:
            ymin, ymax = SpiceUtil.vertical_edges_limits(self.header)
            self.data[:, :, :ymin + 1, :] = np.nan
            self.data[:, :, ymax:, :] = np.nan

        self.uncertainty = None
        self.uncertainty_average = None

        # Building the different wcs depending on the axes
        self.wcs = WCS(self.header)
        self.w_xyt = copy.deepcopy(self.wcs).dropaxis(2)
        self.w_xy = copy.deepcopy(self.w_xyt)
        self.w_xy.wcs.pc[2, 0] = 0
        self.w_xy = self.w_xy.dropaxis(2)
        self.w_spec = copy.deepcopy(self.wcs).sub(["spectral"])




    def return_wavelength_array(self) -> list:
        lenlambda = self.data.shape[1]
        z = np.arange(lenlambda)
        lamb = self.w_spec.world_to_pixel(z)
        return lamb

    def return_wavelength_interval(self) -> list:
        lenlambda = self.data.shape[1]
        z = np.arange(lenlambda)
        lamb = self.w_spec.world_to_pixel(z)
        dlamb = lamb[1] - lamb[0]
        return [lamb[0] - 0.5 * dlamb, lamb[-1] + 0.5 * dlamb]

    def compute_uncertainty(self, verbose: bool = False) -> None:
        if self.uncertainty is not None:
            raise ValueError("uncertainty is already computed")
        av_constant_noise_level, sigma = spice_error(
            data=self.data.to(self.header["BUNIT"]).value, header=self.header, verbose=verbose
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
            If false, the the spectrum is directly taken from the given pixels (or their given coordinates)
        """        
        count_arguments = 0
        error_str_lonlatlims = "lonlat_lims has the wrong format. it should be of the form ((240*u.arcsec, 260*u.arcsec,), (-25*u.arcsec, 25*u.arcsec,))"
        error_str_coords = "coords has the wrong format. it should be a astropy.coordintes.SkyCoord object"
        error_str_pixels = "pixels has the wrong format. it should be a tuple (x, y)"

        for arg in [coords, lonlat_lims, pixels]:
            if arg is not None:
                count_arguments += 1
        if count_arguments != 1:
            raise ValueError("average_spectra_over_region only accepts one argument among coords, lonlat_lims or pixels.")
        
        if coords is not None:
            try :
                if not('Tx' in coords.representation_component_names):
                    raise ValueError("Frame in Skycoord not recognised. Can only use helioprojective frame as of now")
                lon = coords.Tx
                lat = coords.Ty
                x, y = self.w_xy.world_to_pixel(coords)
                x = x.ravel()
                y = y.ravel()
            except:
                raise ValueError(error_str_coords)
            
        if lonlat_lims is not None:
            if not(allow_reprojection):
                raise ValueError("Please allow reprojection if you want to average over a given region in longitude/latitudes limits.")
            try:
                if isinstance(lonlat_lims, tuple) & len(lonlat_lims) == 2 & isinstance(lonlat_lims[0], tuple) & \
                    isinstance(lonlat_lims[0], tuple) & isinstance(lonlat_lims[0][0], u.Quantity):
                    
                    coords_tmp = self.w_xy.pixel_to_world(0, 0)
                    lonlim = lonlat_lims[0]
                    latlim = lonlat_lims[1]
                    coord_tmp = self.w_xy.pixel_to_world([0, 1], [0, 1])
                    ln = coord_tmp.Tx
                    lt = coord_tmp.Ty
                    dlon = ln[1, 1] - ln[0, 0]
                    dlat = lt[1, 1] - lt[0, 0]
                    dlon = dlon.to(ln.unit).value
                    dlat = dlat.to(lt.unit).value
                    lon = np.arange(lonlim[0].to(ln.unit).value, lonlim[1].to(ln.unit).value, dlon)
                    lat = np.arange(latlim[0].to(lt.unit).value, latlim[1].to(lt.unit).value, dlat)
                    

                    coords = SkyCoord(lonlim, latlim, frame= coords_tmp.frame)
                    lon = coords.Tx
                    lat = coords.Ty
                    x, y = self.w_xy.world_to_pixel(coords)
                    x = x.ravel()
                    y = y.ravel()

                else:
                    raise ValueError(error_str_lonlatlims)
            except:
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
            except :
                raise ValueError(error_str_pixels)
        if not(allow_reprojection):
            decx = np.abs(x - np.array(np.round(x), dtype=int))
            decy = np.abs(y - np.array(np.round(y), dtype=int))

            if ((decx < 1.0E-10).all()) & ((decy < 1.0E-10).all()):
                pass
            else:
                raise ValueError("The pixel position you want to average the spectra are not integers. For now, the interpolation of the spectra is not yet implemented")
        

            x = np.array(np.round(x), dtype=int)
            y = np.array(np.round(y), dtype=int)

        lon_mid = np.mean(lon.ravel())
        lat_mid = np.mean(lat.ravel())
        header_av = self.header.copy()
        header_av["CRPIX1"] = 1
        header_av["CRPIX2"] = 1
        header_av["CRVAL1"] = lon_mid.to(header_av["CUNIT1"]).value
        header_av["CRVAL2"] = lat_mid.to(header_av["CUNIT1"]).value
        
        data_av =copy.deepcopy(self.data)
        if allow_reprojection:
            assert data_av_rep.shape[0] == 1
            data_av_rep = np.zeros(data_av.shape[0], data_av.shape[1], len(x))
            for l in range(self.data.shape[1]):
                data_av_rep[0, l, :] = CommonUtil.interpol2d(self.data[0, l, :, :], x=x, y=y, order=1, fill=np.nan,)
            data_av_rep = np.nanmean(data_av_rep, axis=2)
        else:
            data_av = np.nanmean(data_av[:, :, y, x], axis=2)
        data_av = data_av.reshape(data_av.shape[0], data_av.shape[1], 1, 1)

        results = SpiceRasterWindowL2(data=data_av, header=header_av, remove_dumbbells=False)

        if self.uncertainty is None:
            self.compute_uncertainty()
        uncertainty_av = copy.deepcopy(self.uncertainty)
        
        N = len(y)
        for l in ["Signal", "Total"]:
            data_sigma_av = copy.deepcopy(self.uncertainty[l])
            if allow_reprojection:
                assert data_sigma_av.shape[0] == 1
                data_sigma_av = np.zeros(self.uncertainty[l].shape[0], self.uncertainty[l].shape[1], len(x))
                for l in range(self.uncertainty[l].shape[1]):
                    data_sigma_av[0, l, :] = CommonUtil.interpol2d(self.uncertainty[l][0, l, :, :], x=x, y=y, order=1, fill=np.nan,)
                data_sigma_av = (1 / N) * rss(data_sigma_av, axis=2)
            else:
                data_sigma_av = data_sigma_av[:, :, y, x]
                data_sigma_av = (1 / N) * rss(data_sigma_av, axis=2)
            data_sigma_av = data_sigma_av.reshape(data_av.shape[0], data_av.shape[1], 1, 1)
            uncertainty_av[l] = data_sigma_av
 
        results.uncertainty = uncertainty_av

        return results

