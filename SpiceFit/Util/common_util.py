import astropy.io.fits as fits
import astropy.units as u
from astropy.time import Time, TimeDelta
import astropy.constants
import numpy as np
from tqdm import tqdm
from scipy.ndimage import map_coordinates
import astropy.constants


class CommonUtil:

    @staticmethod
    def find_closest_dict_index(utc_to_find, dict_file_reference, threshold_time, time_delay=False,
                                dsun_obs_to_find=None):
        if time_delay:
            if dsun_obs_to_find is None:
                raise ValueError("please enter dsun_obs_ref if time delay is not negligeable.")
            time = np.array([n - TimeDelta(((d * u.m - dsun_obs_to_find) / astropy.constants.c).to("s"))
                             for n, d in zip(dict_file_reference["date-avg"], dict_file_reference["dsun-obs"])],
                            dtype="object")
        else:
            time = dict_file_reference["date-avg"]
        # print(f'{utc_to_find=}')
        # print(f'{time=}')

        delta_time = np.array([np.abs((utc_to_find - n).to(u.s).value) for n in time])
        closest_index = delta_time.argmin()
        delta_time_min = delta_time[closest_index]

        if delta_time_min > threshold_time:
            raise ValueError("Delta time between EUI and SPICE file "
                             "equal to %2f s > %.2f" % (delta_time_min, threshold_time))
        return closest_index, delta_time_min

    @staticmethod
    def find_closest_time(list_to_find: list, list_ref: list, window_to_find=-1, window_ref=-1, time_delay=True,
                          maximal_threshold=15 * u.s):
        """
        Returns a list (of size length(list_to_find) with the index of the closest Fits files.
        Returns
        """
        list_index = np.arr
        for ii, fits_path_to_find in enumerate(tqdm(list_to_find, desc="find_closest_time")):
            with fits.open(fits_path_to_find) as hdul:
                hdu = hdul[window_to_find]
                date_obs_to_find = Time(hdu.header["DATE-AVG"])
                if time_delay:
                    dsun_obs_to_find = hdu.header["DSUN_OBS"]
                time_diff = []
                for jj, fits_path_ref in enumerate(list_ref):
                    with fits.open(fits_path_ref) as hdul_tmp:
                        hdu_tmp = hdul_tmp[window_ref]
                        date_obs_ref = Time(hdu_tmp.header["DATE-AVG"])

                        if time_delay:
                            dsun_obs_ref = hdu_tmp.header["DSUN_OBS"]
                            dt = (np.array(dsun_obs_to_find) - np.array(dsun_obs_ref)) / astropy.constants.c.value
                            date_obs_ref = date_obs_ref + dt * u.s
                        time_diff.append(np.abs((date_obs_to_find - date_obs_ref).to(u.s).value))
                        hdul_tmp.close()
                list_index.append(np.array(time_diff).argmin())
                hdul.close()
            list_index = np.array(list_index, dtype='int')
            selection_error = (list_index > maximal_threshold.to(u.s).value)
            if selection_error.sum() > 0:
                raise ValueError("Threshold delta time of %i s attained" % (maximal_threshold.to(u.s).value))

    @staticmethod
    def ang2pipi(ang):
        """ put angle between ]-180, +180] deg """
        pi = u.Quantity(180, 'deg')
        return - ((- ang + pi) % (2 * pi) - pi)

    @staticmethod
    def interpol2d(image, x, y, fill, order, dst=None):
        """"
        taken from Frederic interpol2d function
        """
        bad = np.logical_or(x == np.nan, y == np.nan)
        x = np.where(bad, -1, x)
        y = np.where(bad, -1, y)

        coords = np.stack((y.ravel(), x.ravel()), axis=0)
        if dst is None:
            dst = np.empty(x.shape, dtype=image.dtype)
        map_coordinates(image, coords, order=order, mode='constant', cval=fill, output=dst.ravel(), prefilter=False)

        return dst

    @staticmethod
    def spatial_resolution(d_au: u.Quantity, alpha=1 * u.arcsec, factor_above_photosphere=1.004):
        fac = factor_above_photosphere
        Rsun = 6.95700E+08 * u.m
        return (d_au - fac * Rsun) * alpha.to("rad").value