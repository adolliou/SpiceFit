import numpy as np
from scipy.ndimage import uniform_filter
import astropy.wcs as wcs
import astropy.units as u
from UtilFunctions.common_util import CommonUtil
from scipy.signal import convolve2d


def rebin_factor(a, factor, function):
    """
    Parameters
    ----------
    a: np.array
        Array to rebin
    factor: tuple
        Rebin factor, on each dimension
    function: function
        Function to apply on each superpixel

    Rebin by some factor on each dimension, by applying some function to reduce
    pixel values into macropixel values
    function must have an axis kwarg, like np.sum. np.mean...
    """
    if np.any(np.mod(np.array(a.shape, dtype=int), np.array(factor, dtype=int))):
        print(f"{factor=}")
        print(f"{a.shape=}")
        raise ValueError("modulos not null")
    ndim = a.ndim
    compression_pairs = [(s // f, f) for (s, f) in zip(a.shape, factor)]
    # print(f'{compression_pairs=}')
    flattened = [l for p in compression_pairs for l in p]
    # print(f'{flattened=}')
    a = a.reshape(flattened)
    axes = tuple(1 + 2 * i for i in range(ndim))
    # print(f'{axes=}')
    return function(a, axis=axes)

class Binning:

    @staticmethod
    def downsampling_uniform_filter(data: np.array, size: tuple) -> np.array:
        output = np.empty_like((data))
        uniform_filter(input=data, size=size, output=output, mode="reflect", )
        return output

    @staticmethod
    def downsampling_convolve2d(data: np.array, size: tuple) -> np.array:
        output = np.empty_like((data))
        kernel = np.ones(size, dtype=np.float64)

        convolve2d(in1=data, in2=kernel, mode="same", boundary="symm")
        return output

    @staticmethod
    def binned_coordinates(wcs_xy: wcs.WCS, size: tuple):
        idx_lon = np.where(np.array(wcs_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
        idx_lat = np.where(np.array(wcs_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
        idx_utc = np.where(np.array(wcs_xy.wcs.ctype, dtype="str") == "UTC")[0][0]

        x, y, t = np.meshgrid(np.arange(wcs_xy.pixel_shape[idx_lon]), np.arange(wcs_xy.pixel_shape[idx_lat]),
                              np.arange(wcs_xy.pixel_shape[idx_utc]), )

        x_new = np.array([
            (x[yy * (size[0]), xx * (size[1]), tt]
             + x[yy * (size[0]) + size[0], xx * (size[1]) + size[1]], tt) / 2 for
            xx in range(int(len(x) - 1) / (size[1])) for yy in range(int(len(y) - 1) / (size[0]))
            for tt in range(int(len(t) - 1))
        ], dtype=np.float64)

        y_new = np.array([
            (y[yy * (size[0]), xx * (size[1])] + y[yy * (size[0]) + size[0], xx * (size[1]) + size[1]]) / 2 for
            xx in range(int(len(x) - 1) / (size[1])) for yy in range(int(len(y) - 1) / (size[0]))
        ], dtype=np.float64)

        y_new = np.array([
            (y[yy * (size[0]), xx * (size[1])] + y[yy * (size[0]) + size[0], xx * (size[1]) + size[1]]) / 2 for
            xx in range(int(len(x) - 1) / (size[1])) for yy in range(int(len(y) - 1) / (size[0]))
        ], dtype=np.float64)

        return x_new, y_new

    def binning(self, data_yx: np.array, wcs_xy: wcs.WCS, size: tuple):
        """"
        for now, just do kernels with odd shapes, or the fits will be displaced
        """
        data_binned = self.downsampling_uniform_filter(data=data_yx, size=size)
        x_new, y_new = self.binned_coordinates(wcs_xy=wcs_xy, size=size)
        CommonUtil.interpol2d(image=data_binned, x=x_new, y=y_new, fill=-32762, dst=data_binned, order=1)

        data_binned[data_binned == -32762] = np.nan

        longitude_new, latitude_new = wcs_xy.pixel_to_world(x_new, y_new)
        pad_longitude_new = longitude_new[0, 0] - longitude_new[1, 0]
        pad_latitude_new = latitude_new[0, 0] - latitude_new[0, 1]

        wcs_new = wcs.WCS(naxis=2)
        longitude_new_1d = np.arange(
            longitude_new.min() - pad_longitude_new,
            longitude_new.max() + longitude_new.shape[0] + pad_longitude_new,
            longitude_new.shape[0],
        )
        latitude_new_1d = np.arange(
            latitude_new.min() - pad_latitude_new,
            latitude_new.max() + latitude_new.shape[1] + pad_latitude_new,
            latitude_new.shape[1],
        )

        longitude_new, latitude_new = np.meshgrid(longitude_new_1d, latitude_new_1d)
        wcs_new.wcs.cdelt = [pad_longitude_new, pad_latitude_new]
        wcs_new.wcs.crpix = [longitude_new.shape[0] // 2, latitude_new.shape[1] // 2]
        wcs_new.wcs.crval = [longitude_new_1d[int(wcs_new.wcs.crpix[0])],
                             latitude_new_1d[int(wcs_new.wcs.crpix[1])]]
        wcs_new.wcs_ctype = ["HPLN-TAN", "HPLT-TAN"]
        wcs_new.wcs.cunit = ["arcsec", "arcsec"]
