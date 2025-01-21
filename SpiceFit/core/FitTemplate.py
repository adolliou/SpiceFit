import astropy.units as u
from yaml import safe_load


class FitTemplate:

    def __init__(self, filename=None, parinfo=None, **kwargs) -> None:
        """

        :param filename: (optional)
        :param parinfo: dictionary, should have the same keys as a parinfo dictionary
        :param kwargs: additional parameters to modify or add to parinfo
        """
        self._parinfo = {"fit": {}, "info": {}}

        if (filename is not None) & (parinfo is None):
            if isinstance(filename, str):
                self.filename = filename
                self._parinfo = {}
                with open(self.filename, 'r') as f:
                    self._parinfo = safe_load(f)
            else:
                raise TypeError("filename must be a string or None")
        elif (filename is None) & (parinfo is not None):
            self.filename = "unknown file"
            test_parinfo = self.check_parinfo(parinfo=parinfo)
            if not test_parinfo:
                raise TypeError("parinfo dictionary must have the valid format")
            self._parinfo = parinfo
        else:
            raise ValueError("must either have filename or parinfo as input")

        # else:
        #     for key in kwargs:
        #         if key in [
        #             "type",
        #             "name",
        #             "n_components",
        #             "guess",
        #             "units",
        #             "max_arr",
        #             "min_arr",
        #             "trans_a",
        #             "trans_b",
        #         ]:
        #             self._parinfo["fit"][key] = kwargs[key]
        #         elif key in [
        #             "name",
        #             "elem",
        #             "ion",
        #             "wave",
        #             "unit_wave",
        #             "lvl_low",
        #             "lvl_up"
        #         ]:
        #             self._parinfo["info"][key] = kwargs[key]
        test_parinfo = self.check_parinfo()
        if not test_parinfo:
            raise TypeError("the keyword arguments must have the valid format")

        self.central_wave = u.Quantity(self._parinfo["fit"]["guess"][1], "angstrom")
        self.wave_interval = [
            u.Quantity(self._parinfo["fit"]["min_arr"][1], "angstrom"),
            u.Quantity(self._parinfo["fit"]["max_arr"][1], "angstrom"),
        ]

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
            raise ValueError(f"template must be a dictionary")
        if (
                ((len(test.keys()) != 2) and (len(test.keys()) != 3))
                | ("info" not in test.keys())
                | ("fit" not in test.keys())
        ):
            raise ValueError(f"template must be a dictionary with 2 or 3 keys")
        if type(test["info"]) is not dict:
            raise ValueError(f"template['info'] must be a dictionary")
        keys = ["name",
                "elem",
                "ion",
                "wave",
                "unit_wave",
                "lvl_low",
                "lvl_up"]
        formats = [str,
                   str,
                   int,
                   float,
                   str,
                   str,
                   str]
        for key, format_ in zip(keys, formats):
            if key not in test["info"].keys():
                raise ValueError(f"key {key} must exist")
            elif type(test["info"][key]) is not format_:
                raise ValueError(f"key {key} must be in {format_}")

        if type(test["fit"]) is not dict:
            raise ValueError(f"template['fit'] must be a dictionary")

        keys = [
            "type",
            "name",
            "n_components",
            "guess",
            "units",
            "max_arr",
            "min_arr",
            "trans_a",
            "trans_b",
        ]
        formats = [
            str,
            list,
            list,
            list,
            list,
            list,
            list,
            list,
            list,
        ]
        for key, format_ in zip(keys, formats):
            if key not in test["fit"].keys():
                raise ValueError(f"key {key} must exist ")
            elif type(test["info"][key]) is not format_:
                raise ValueError(f"key {key} must be in {format_}")

            if format_ is list:
                for sub in test["fit"][key]:
                    if type(sub) is not list:
                        raise ValueError(f"{key} a list")
        if (len(test.keys()) == 3) and ("special_instructions" not in test.keys()):
            raise ValueError(f"Fittemplate third key must be 'special_instructions'")
        return True
