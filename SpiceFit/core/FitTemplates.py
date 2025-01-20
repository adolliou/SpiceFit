import astropy.units as u


class FitTemplate:

    def __init__(self, filename=None, parinfo=None, **kwargs) -> None:
        """

        :param filename: (optional)
        :param parinfo: dictionary, should have the same keys as a parinfo dictionary
        :param kwargs: additional parameters to modify or add to parinfo
        """

        if filename is None:
            self.filename = "unknown file"
        elif isinstance(filename, str):
            self.filename = filename
        if type(filename) is not str:
            raise TypeError("filename must be a string")
        if parinfo is not None:
            test_parinfo = self.check_parinfo(parinfo=parinfo)
            if not test_parinfo:
                raise TypeError("parinfo dictionary must have the valid format")
            self._parinfo = parinfo
        else:
            self._parinfo = {'fit': {}, 'info': {}}
        for key in kwargs:
            if key in ['type', 'guess', 'max_arr', 'min_arr', 'trans_a', 'trans_b']:
                self._parinfo['fit'][key] = kwargs[key]
            elif key in ['name', 'elem', 'ion', 'wave', 'lvl_low', 'lvl_up']:
                self._parinfo['info'][key] = kwargs[key]
        test_parinfo = self.check_parinfo()
        if not test_parinfo:
            raise TypeError("the keyword arguments must have the valid format")

        self.central_wave = u.Quantity(self._parinfo['fit']['guess'][1], "angstrom")
        self.wave_interval = [u.Quantity(self._parinfo['fit']['min_arr'][1], "angstrom"),
                              u.Quantity(self._parinfo['fit']['max_arr'][1], "angstrom")]

    @property
    def parinfo(self):
        """
        dictionary containing information about the lines divided into the 'info' and 'fit' categories.
        :return:
        """
        return self._parinfo

    @parinfo.setter
    def parinfo(self, parinfo: dict):
        """
        Set the _parinfo dictionnary, while verifying that every key and value are consistent.
        :param parinfo:
        :return:
        """
        if not self.check_parinfo(parinfo=parinfo):
            raise TypeError("parinfo dictionary must have the valid format")
        self._parinfo = parinfo

    def check_parinfo(self, parinfo=None):
        """
        Check if parinfo or self._parinfo has the valid format.
        :param parinfo: dictionnary containing information about the lines divided into the 'info' and 'fit' categories.
        :return:
        """
        if parinfo is not None:
            test = parinfo
        else:
            test = self.parinfo

        if type(test) is not dict:
            return False
        if ((len(test.keys()) != 2) and (len(test.keys()) != 3)) | ("info" not in test.keys()) | ("fit" not in test.keys()):
            return False
        if type(test["info"]) is not dict:
            return False
        keys = ['name', 'elem', 'ion', 'wave', 'lvl_low', 'lvl_up']
        formats = [str, str, int, float, str, str]
        for key, format_ in zip(keys, formats):
            if (key not in test["info"].keys()) | (type(key) is not format_):
                return False

        if type(test["fit"]) is not dict:
            return False
        keys = ['type', 'guess', 'max_arr', 'min_arr', 'trans_a', 'trans_b']
        formats = [str, list, list, list, list, list, list]
        for key, format_ in zip(keys, formats):
            if (key not in test["fit"].keys()) | (type(key) is not format_):
                return False
            if format_ is list:
                for sub in test["fit"][key]:
                    if (type(sub) is not int) and (type(sub) is not float):
                        return False
        if (len(test.keys()) == 3) and ("special_instructions" not in test.keys()):
            return False
        return True
