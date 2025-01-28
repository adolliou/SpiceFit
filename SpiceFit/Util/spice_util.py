import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

class SpiceUtil:

    @staticmethod
    def slit_pxl(header):

        """ Compute the first and last pixel of the slit from a FITS header """
        ybin = header['NBIN2']
        h_detector = 1024 / ybin
        if header['DETECTOR'] == 'SW':
            h_slit = 600 / ybin
        elif header['DETECTOR'] == 'LW':
            h_slit = 626 / ybin
        else:
            raise ValueError(f"unknown detector: {header['DETECTOR']}")
        slit_beg = (h_detector - h_slit) / 2
        slit_end = h_detector - slit_beg
        slit_beg = slit_beg - header['PXBEG2'] / ybin + 1
        slit_end = slit_end - header['PXBEG2'] / ybin + 1
        slit_beg = int(np.ceil(slit_beg))
        slit_end = int(np.floor(slit_end))
        return slit_beg, slit_end

    @staticmethod
    def vertical_edges_limits(header):
        iymin, iymax = SpiceUtil.slit_pxl(header)
        iymin += int(20 / header['NBIN2'])
        iymax -= int(20 / header['NBIN2'])
        return iymin, iymax

    @staticmethod
    def create_intensity_map(path_to_l3, index_amplitude=0, index_width=2):
        hdul = fits.open(path_to_l3)
        hdu_results = hdul[0]
        data = hdu_results.data.copy()
        hdr = hdu_results.header.copy()
        hdul.close()

        missing = hdr["ANA_MISS"]
        condition_missing = np.array((data[:, :, index_amplitude] == missing) | (data[:, :, index_width] == missing),
                                     dtype="bool")

        intensity = data[:, :, index_amplitude] * data[:, :, index_width] * np.sqrt(2 * np.pi)
        intensity[condition_missing] = np.nan

        return hdr, intensity

    @staticmethod
    def _data_treatment_spice(self, hdr, data):
        w_small = WCS(hdr)
        w2 = w_small.deepcopy()
        w2.wcs.pc[3, 0] = 0
        w2.wcs.pc[3, 1] = 0
        w_xyt = w2.dropaxis(2)
        # TODO 1) select slit positions 2) correct for solar rotation
        w_xy_small = w_xyt.dropaxis(2)
        return w_xy_small

    @staticmethod
    def extract_spice_coordinates_l3(hdr, return_type='xy'):
        w_small = WCS(hdr)
        w2 = w_small.deepcopy()

        w2.wcs.pc[3, 0] = 0

        if return_type == 'xy':
            w2.wcs.pc[3, 1] = 0
            w_xyt = w2.dropaxis(0)
            w_xy = w_xyt.dropaxis(2)

            idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            x_small, y_small = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                                           np.arange(w_xy.pixel_shape[idx_lat]), )
            longitude_small, latitude_small = w_xy.pixel_to_world(x_small, y_small)
            return longitude_small, latitude_small
        elif return_type == 'xyt':
            w_xyt = w2.dropaxis(0)

            idx_lon = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            idx_utc = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "UTC")[0][0]
            x_small, y_small, z_small = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]),
                                                    np.arange(w_xyt.pixel_shape[idx_lat]),
                                                    np.arange(w_xyt.pixel_shape[idx_utc]), )
            longitude_small, latitude_small, utc_small = w_xyt.pixel_to_world(x_small, y_small, z_small)
            return longitude_small, latitude_small, utc_small

    @staticmethod
    def extract_spice_coordinates_l2(hdr, return_type='xy'):
        w = WCS(hdr)
        w_xyt = w.dropaxis(2)

        if return_type == 'xy':
            w_xyt.wcs.pc[2, 0] = 0
            w_xy = w_xyt.dropaxis(2)
            idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            x_small, y_small = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                                           np.arange(w_xy.pixel_shape[idx_lat]), )
            longitude_small, latitude_small = w_xy.pixel_to_world(x_small, y_small)
            return longitude_small, latitude_small
        elif return_type == 'xyt':
            idx_lon = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            idx_utc = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "UTC")[0][0]
            x_small, y_small, z_small = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]),
                                                    np.arange(w_xyt.pixel_shape[idx_lat]),
                                                    np.arange(w_xyt.pixel_shape[idx_utc]), )
            longitude_small, latitude_small, utc_small = w_xyt.pixel_to_world(x_small, y_small, z_small)
            return longitude_small, latitude_small, utc_small

    @staticmethod
    def recenter_crpix_in_header_L2(hdr):
        w = WCS(hdr)
        w_xyt = w.dropaxis(2)

        if "ZNAXIS1" in hdr:
            naxis1 = hdr["ZNAXIS1"]
            naxis2 = hdr["ZNAXIS2"]
            naxis3 = hdr["ZNAXIS3"]
        else:
            naxis1 = hdr["NAXIS1"]
            naxis2 = hdr["NAXIS2"]
            naxis3 = hdr["NAXIS3"]

        x_mid = (naxis1 - 1) / 2
        y_mid = (naxis2 - 1) / 2
        t_mid = (naxis3 - 1) / 2
        coords = w_xyt.pixel_to_world(np.array([x_mid]), np.array([y_mid]), np.array([t_mid]))
        lon_mid = coords[0].Tx
        lat_mid = coords[0].Ty

        lon_mid = lon_mid[0].to(hdr["CUNIT1"]).value
        lat_mid = lat_mid[0].to(hdr["CUNIT2"]).value

        hdr["CRVAL1"] = lon_mid
        hdr["CRVAL2"] = lat_mid
        hdr["CRPIX1"] = (naxis1 + 1) / 2
        hdr["CRPIX2"] = (naxis2 + 1) / 2

    @staticmethod
    def extract_l3_data(path_spice: str, line: dict, index_line: int, window=0):

        with fits.open(path_spice) as hdul_spice:
            hdu = hdul_spice[window]
            data = hdu.data
            data_l3 = {"amplitude": data[:, :, line["amplitude"][index_line]],
                       "width": data[:, :, line["width"][index_line]],
                       "chi2": data[:, :, line["chi2"][index_line]],
                       "background": data[:, :, line["background"][index_line]],
                       "lambda": data[:, :, line["lambda"][index_line]]}
            data_l3["chi2"] = np.where(data_l3["amplitude"] == hdu.header["ANA_MISS"], np.nan, data_l3["chi2"])

            for key in ["amplitude", "width", "background", "lambda"]:
                data_l3[key] = np.where(data_l3["chi2"] == 0, np.nan, data_l3[key])
                data_l3[key] = np.where(data_l3[key] == hdu.header["ANA_MISS"], np.nan,
                                        data_l3[key])

            data_l3["radiance"] = data_l3["amplitude"] * data_l3["width"] * np.sqrt(2 * np.pi) * 0.424660900

            return data_l3
