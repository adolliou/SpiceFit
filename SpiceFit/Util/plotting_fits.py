from matplotlib import pyplot as plt
import numpy as np
from .common_util import CommonUtil
from astropy.wcs import WCS
import astropy.units as u
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.colors as colors

from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, SqrtStretch, LinearStretch, LogStretch


class CmapUtil:

    @staticmethod
    def create_cdict(r, g, b):
        """
        Create the color tuples in the correct format.
        """
        i = np.linspace(0, 1, r.size)
        cdict = {name: list(zip(i, el / 255.0, el / 255.0))
                 for el, name in [(r, 'red'), (g, 'green'), (b, 'blue')]}
        return cdict

    @staticmethod
    def get_idl3(path_idl3: str):
        # The following values describe color table 3 for IDL (Red Temperature)
        return np.loadtxt(path_idl3, delimiter=',')

    @staticmethod
    def _cmap_from_rgb(r, g, b, name):
        cdict = CmapUtil.create_cdict(r, g, b)
        return colors.LinearSegmentedColormap(name, cdict)

    @staticmethod
    def create_aia_wave_dict(path_idl3: str):
        idl_3 = CmapUtil.get_idl3(path_idl3)
        r0, g0, b0 = idl_3[:, 0], idl_3[:, 1], idl_3[:, 2]

        c0 = np.arange(256, dtype='f')
        c1 = (np.sqrt(c0) * np.sqrt(255.0)).astype('f')
        c2 = (np.arange(256) ** 2 / 255.0).astype('f')
        c3 = ((c1 + c2 / 2.0) * 255.0 / (c1.max() + c2.max() / 2.0)).astype('f')

        aia_wave_dict = {
            1600 * u.angstrom: (c3, c3, c2),
            1700 * u.angstrom: (c1, c0, c0),
            4500 * u.angstrom: (c0, c0, b0 / 2.0),
            94 * u.angstrom: (c2, c3, c0),
            131 * u.angstrom: (g0, r0, r0),
            171 * u.angstrom: (r0, c0, b0),
            193 * u.angstrom: (c1, c0, c2),
            211 * u.angstrom: (c1, c0, c3),
            304 * u.angstrom: (r0, g0, b0),
            335 * u.angstrom: (c2, c0, c1)
        }
        return aia_wave_dict

    @staticmethod
    def solohri_lya1216_color_table(path_idl3: str):
        solohri_lya1216 = CmapUtil.get_idl3(path_idl3)
        solohri_lya1216[:, 2] = solohri_lya1216[:, 0] * np.linspace(0, 1, 256)
        return CmapUtil._cmap_from_rgb(*solohri_lya1216.T, 'SolO EUI HRI Lyman Alpha')

    @staticmethod
    def aia_color_table(wavelength: u.angstrom, path_idl3: str):
        """
        Returns one of the fundamental color tables for SDO AIA images.

        Based on aia_lct.pro part of SDO/AIA on SSWIDL written by Karel
        Schrijver (2010/04/12).

        Parameters
        ----------
        wavelength : `~astropy.units.quantity`
            Wavelength for the desired AIA color table.
        """
        aia_wave_dict = CmapUtil.create_aia_wave_dict(path_idl3)
        try:
            r, g, b = aia_wave_dict[wavelength]
        except KeyError:
            raise ValueError("Invalid AIA wavelength. Valid values are "
                             "1600,1700,4500,94,131,171,193,211,304,335.")

        return CmapUtil._cmap_from_rgb(r, g, b, 'SDO AIA {:s}'.format(str(wavelength)))


class PlotFits:
    @staticmethod
    def get_range(data, stre='log', imax=99.5, imin=2):
        """
        :param data:
        :param stretch: 'sqrt', 'log', or 'linear' (default)
        :return: norm
        """
        if np.isnan(data).sum() == data.size:
            return None

        isnan = np.isnan(data)
        data = data[~isnan]
        do = False
        if imax > 100:
            vmin, vmax = AsymmetricPercentileInterval(imin, 100).get_limits(data)
            vmax = vmax * imax/100
        else:
            vmin, vmax = AsymmetricPercentileInterval(imin, imax).get_limits(data)

        #    print('Vmin:', vmin, 'Vmax', vmax)
        if stre is None:
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif stre == 'sqrt':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        elif stre == 'log':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            raise ValueError('Bad stre value: either None or sqrt')
        return norm
    @staticmethod
    def plot_fov_rectangle(data, slc=None, path_save=None, show=True, plot_colorbar=True, norm=None, angle=0):
        fig = plt.figure()
        ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        PlotFits.plot_fov(data=data, show=False, fig=fig, ax=ax, norm=norm)
        rect = patches.Rectangle((slc[1].start, slc[0].start),
                                 slc[1].stop - slc[1].start, slc[0].stop - slc[0].start, linewidth=1,
                                 edgecolor='r', facecolor='none', angle=angle)
        ax.add_patch(rect)
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)

    @staticmethod
    def plot_fov(data, slc=None, path_save=None, show=True, plot_colorbar=True, fig=None, ax=None, norm=None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        if slc is not None:
            im = ax.imshow(data[slc[0], slc[1]], origin="lower", interpolation="none", norm=norm)
        else:
            im = ax.imshow(data, origin="lower", interpolation="None", norm=norm)
        if plot_colorbar:
            fig.colorbar(im, label="DN/s")
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
    @staticmethod
    def build_regular_grid(longitude, latitude, lonlims=None, latlims=None):
        x = np.abs((longitude[0, 1] - longitude[0, 0]).to("deg").value)
        y = np.abs((latitude[0, 1] - latitude[0, 0]).to("deg").value)
        dlon = np.sqrt(x**2 + y**2)

        x = np.abs((longitude[1, 0] - longitude[0, 0]).to("deg").value)
        y = np.abs((latitude[1, 0] - latitude[0, 0]).to("deg").value)
        dlat = np.sqrt(x**2 + y**2)

        longitude1D = np.arange(np.min(CommonUtil.ang2pipi(longitude).to(u.deg).value),
                                np.max(CommonUtil.ang2pipi(longitude).to(u.deg).value), dlon)
        latitude1D = np.arange(np.min(CommonUtil.ang2pipi(latitude).to(u.deg).value),
                               np.max(CommonUtil.ang2pipi(latitude).to(u.deg).value), dlat)
        if (lonlims is not None) or (latlims is not None):
            longitude1D = longitude1D[(longitude1D > CommonUtil.ang2pipi(lonlims[0]).to("deg").value) &
                                      (longitude1D < CommonUtil.ang2pipi(lonlims[1]).to("deg").value)]
            latitude1D = latitude1D[(latitude1D > CommonUtil.ang2pipi(latlims[0]).to("deg").value) &
                                    (latitude1D < CommonUtil.ang2pipi(latlims[1]).to("deg").value)]
        longitude_grid, latitude_grid = np.meshgrid(longitude1D, latitude1D)

        longitude_grid = longitude_grid * u.deg
        latitude_grid = latitude_grid * u.deg
        dlon = dlon * u.deg
        dlat = dlat * u.deg
        return longitude_grid, latitude_grid, dlon, dlat

    @staticmethod
    def extend_regular_grid(longitude_grid, latitude_grid, delta_longitude, delta_latitude):
        x = np.abs((longitude_grid[0, 1] - longitude_grid[0, 0]).to("deg").value)
        y = np.abs((latitude_grid[0, 1] - latitude_grid[0, 0]).to("deg").value)
        dlon = np.sqrt(x**2 + y**2)

        x = np.abs((longitude_grid[1, 0] - longitude_grid[0, 0]).to("deg").value)
        y = np.abs((latitude_grid[1, 0] - latitude_grid[0, 0]).to("deg").value)
        dlat = np.sqrt(x**2 + y**2)

        delta_longitude_deg = CommonUtil.ang2pipi(delta_longitude).to("deg").value
        delta_latitude_deg = CommonUtil.ang2pipi(delta_latitude).to("deg").value

        longitude1D = np.arange(np.min(CommonUtil.ang2pipi(longitude_grid).to(u.deg).value - 0.5 * delta_longitude_deg),
                                np.max(CommonUtil.ang2pipi(longitude_grid).to(u.deg).value) + 0.5 * delta_longitude_deg,
                                dlon)
        latitude1D = np.arange(np.min(CommonUtil.ang2pipi(latitude_grid).to(u.deg).value - 0.5 * delta_latitude_deg),
                               np.max(CommonUtil.ang2pipi(latitude_grid).to(u.deg).value) + 0.5 * delta_latitude_deg,
                               dlat)

        longitude_grid_ext, latitude_grid_ext = np.meshgrid(longitude1D, latitude1D)
        longitude_grid_ext = longitude_grid_ext * u.deg
        latitude_grid_ext = latitude_grid_ext * u.deg

        return longitude_grid_ext, latitude_grid_ext


