import astropy.units as u
import numpy as np
from yaml import safe_load
from ..util.constants import Constants
import copy
import re
import string


class FittingModel:

    bound_equation_re = re.compile(
        r'''
    (?P<ref>(guess))
    _(?P<operation>(plus|minus|times))
    _(?P<value>\d*\.\d*)
    ''',
        re.VERBOSE)


    constrain_equation_re = re.compile(
        r'''
    (?P<ref>(.+?))
    _(?P<operation>(plus|minus|times))
    _(?P<value>\d*\.\d*)
    ''',
        re.VERBOSE)


    def __init__(self, filename=None, parinfo=None, ) -> None:
        """

        :param filename: (optional)
        :param parinfo: dictionary, should have the same keys as a parinfo dictionary
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
            test_parinfo = self._check_parinfo(parinfo=parinfo)
            if not test_parinfo:
                raise TypeError("parinfo dictionary must have the valid format")
            self._parinfo = parinfo
        else:
            raise ValueError("must either have filename or parinfo as input")

        test_parinfo = self._check_parinfo()
        if not test_parinfo:
            raise TypeError("the keyword arguments must have the valid format")

        # Format of self.free_params["gaussian"][0]["I"/"x"/"s"]["free"/"constrain"/"guess"/"bounds"/"unit"]
        self._params_all = self._build_self_params_all()
        self._params_free = None
        # Format of self.free_params["gaussian"][0]["I"/"x"/"s"]["notation"/"guess"/"bounds"/"unit"]

        # self.central_wave = u.Quantity(self._parinfo["fit"]["guess"][1], "angstrom")
        # self.wave_interval = [
        #     u.Quantity(self._parinfo["fit"]["min_arr"][1], "angstrom"),
        #     u.Quantity(self._parinfo["fit"]["max_arr"][1], "angstrom"),
        # ]
        #
        # self.component_name = self.parinfo["fit"]["name"]
        # self.main_line = self.parinfo["main_line"]


    @property
    def params_free(self):
        """
        getter of the free_params dictionary, containing all the informations about the free parameters of the function.
        format : self.free_params["gaussian"][0]["I"/"x"/"s"]["notation"/"guess"/"bounds"/"unit/"type_constrain"]
        "notation" (str) : how the coefficient appears in the fitting and jacobian functions, must be unique ("I1")
        "guess" : (float64) guess value in "unit"
        "bounds" : (tuple[float64, float64]) bounds tuple value in "unit"
        "unit" : must be consistent with

        :return:
        # """
        # raise ValueError("Must not access free parameters directly. "
        #                  "Please generate them through self.generate_free_params()")
        self._params_free = self._generate_empty_params_free_dict()
        type_list, index_list, coeff_list = self._gen_mapping_params()
        for type_, idx_, coeff_ in zip(type_list, index_list, coeff_list):

            d = copy.deepcopy(self.params_all[type_][idx_][coeff_]["bounds"])
            bounds = []

            for ii, bbb in enumerate(d):
                if type(bbb) is dict:
                    if bbb["ref"] == "guess":
                        value = float(bbb["value"])
                        if bbb["operation"] == "plus":
                            bounds.append(np.float64(self.params_all[type_][idx_][coeff_]["guess"] + value))
                        elif bbb["operation"] == "minus":
                            bounds.append(np.float64(self.params_all[type_][idx_][coeff_]["guess"] - value))
                        elif bbb["operation"] == "times":
                            bounds.append(np.float64(self.params_all[type_][idx_][coeff_]["guess"] * value))
                    else:
                        raise NotImplementedError("unknown guess")
                else:
                    bounds.append(np.float64(bbb))


            self._params_free[type_][idx_][coeff_] = {
                "guess": self.params_all[type_][idx_][coeff_]["guess"],
                "bounds": bounds,
                "unit": self.params_all[type_][idx_][coeff_]["unit"],

            }

        return self._params_free

    @params_free.setter
    def params_free(self, params_free: dict):
        """
        Setter of the free params dictionary, containing all the informations about the free parameters of the function.
        Must change the self._all_params dictionnary for the contrained coefficients, according to their constraints.
        :param params_free:
        """
        raise ValueError("Must not modify free parameters directly. "
                         "Please generate them through self.generate_free_params()")
        # self.params_free = params_free

    @property
    def params_all(self):
        """
        self.params_all getter.
        """
        return self._params_all

    @params_all.setter
    def params_all(self, params_all: dict):

        self._params_all = params_all


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
        if not self._check_parinfo(parinfo=parinfo):
            raise TypeError("parinfo dictionary must have the valid format")
        self._parinfo = parinfo

    def get_component_info(self, component_name):
        if self.parinfo is None:
            raise ValueError("parinfo dictionary must have the valid format")

        name_info = [n["name"] for n in self.parinfo["info"]]
        index = np.where(np.array(name_info) == component_name)[0]
        if len(index) > 1:
            raise ValueError("Duplicated names in info list.")
        if len(index) == 0:
            raise ValueError("Component name not in info list.")
        return self.parinfo["info"][index[0]]

    def get_component_fit(self, component_name):
        if self.parinfo is None:
            raise ValueError("parinfo dictionary must have the valid format")

        name_fit = self.parinfo["fit"]["name"]
        index = np.where(np.array(name_fit) == component_name)[0]
        if len(index) > 1:
            raise ValueError("Duplicated names in info list.")
        if len(index) == 0:
            raise ValueError("Component name not in fit list.")
        fit_info_dict = {}
        for key in self.parinfo["fit"].key():
            fit_info_dict[key] = self.parinfo["fit"][key][index[0]]
        return fit_info_dict

    def _generate_empty_params_free_dict(self):
        _params_all = {}
        _params_all = copy.deepcopy(self.params_all)

        type_list, index_list, coeff_list = self._gen_mapping_params()
        for type_, idx_, coeff_ in zip(type_list, index_list, coeff_list):

            for key in list(_params_all[type_][idx_][coeff_].keys()):
                del _params_all[type_][idx_][coeff_][key]

            _params_all[type_][idx_][coeff_][key] = {
                "guess": None,
                "bounds": None,
                "unit": None,
            }
        return _params_all


    def _build_self_params_all(self):

        type_list = self.parinfo["fit"]["type"]
        name_list = self.parinfo["fit"]["name"]
        n_components_list = self.parinfo["fit"]["n_components"]
        guess_list = self.parinfo["fit"]["guess"]
        coeff_type_list = self.parinfo["fit"]["coeff_type"]
        units_list = self.parinfo["fit"]["units"]
        max_arr_list = self.parinfo["fit"]["max_arr"]
        min_arr_list = self.parinfo["fit"]["min_arr"]
        trans_a_list = self.parinfo["fit"]["trans_a"]
        trans_b_list = self.parinfo["fit"]["trans_b"]

        params_all = {}
        index_gaussian = 0
        index_polynomial= 0

        for ii, type in enumerate(type_list):
            if type not in params_all.keys():
                params_all[type] = []

            if type == "gaussian":
                params_all[type].append({})
                params_all[type][-1] = {
                    "I": {},
                    "x": {},
                    "s": {},
                }
                coeff_type = coeff_type_list[ii]
                idxi = np.where(np.array(coeff_type) == "I")[0]
                idxx = np.where(np.array(coeff_type) == "lambda")[0]
                idxs = np.where(np.logical_or(np.array(coeff_type) == "fwhm", np.array(coeff_type) == "sigma"))[0]

                for idx in  [idxi, idxx, idxs]:
                    if len(idx) != 1:
                        raise ValueError("coeff_type index must be with in the format (I, lambda, fwhm|sigma) "
                                         "for a Gaussian function")
                indexes_in_yaml = [idxi[0], idxx[0], idxs[0]]
                coeffs = ["I", "x", "s"]

                index = copy.deepcopy(index_gaussian)
                index_gaussian = index_gaussian + 1

            elif type == "polynomial":

                ncomp = n_components_list[ii]
                if ncomp > 10:
                    raise ValueError("Polynom power too large above 10")

                params_all[type].append({})

                coeffs = list(string.ascii_lowercase)[:ncomp]
                for n in coeffs:
                    params_all[type][-1][n] = {}
                indexes_in_yaml = range(ncomp)

                index = copy.deepcopy(index_polynomial)
                index_polynomial = index_polynomial + 1
            else:
                raise NotImplementedError("only gaussian and polynomial types are implemented")

            for idx, coeff in zip(indexes_in_yaml, coeffs):


                _guess = guess_list[ii][idx]
                _unit = units_list[ii][idx]
                _max_arr = max_arr_list[ii][idx]
                _min_arr = min_arr_list[ii][idx]
                _trans_a = trans_a_list[ii][idx]
                _trans_b = trans_b_list[ii][idx]
                _guess = (u.Quantity(_guess, _unit) * _trans_a + u.Quantity(_trans_b, _unit))
                guess = self._transform_to_conventional_unit(_guess)
                _bounds_to_parse = [_min_arr, _max_arr]
                bounds = self._parse_bounds(bounds_to_parse=_bounds_to_parse,
                                            unit=_unit,trans_a = _trans_a, trans_b = _trans_b)

                notation = f'{str(coeff)}{ii:d}'

                params_all[type][-1][coeff] = {
                    "guess": guess.value,
                    "bounds": bounds,
                    "unit": str(guess.unit),
                    "notation": notation,
                    "index": index,
                    "free": True,
                    "type_constrain": None,

                    "index_in_fit_yaml": [ii, idx]
                }

        self.params_all = params_all
        if "constrain_lines" in self.parinfo.keys():
            _type_list, _index_list, _coeff_list = self._gen_mapping_params()

            type_constrain_list = self.parinfo["constrain_lines"]["type_constrain"]
            constrained_list = self.parinfo["constrain_lines"]["constrained"]

            for _type, _index, _coeff in zip(_type_list, _index_list, _coeff_list):

                A = params_all[_type][_index][_coeff]
                index_in_fit_yaml = A["index_in_fit_yaml"]
                _type_constrain = type_constrain_list[index_in_fit_yaml[0]][index_in_fit_yaml[1]]
                _constrained = constrained_list[index_in_fit_yaml[0]][index_in_fit_yaml[1]]
                _trans_a = trans_a_list[index_in_fit_yaml[0]][index_in_fit_yaml[1]]
                _trans_b = trans_b_list[index_in_fit_yaml[0]][index_in_fit_yaml[1]]
                _unit = units_list[index_in_fit_yaml[0]][index_in_fit_yaml[1]]

                if _constrained:
                    type_constraint = self._parse_constrains_(constrain_to_parse=_type_constrain,
                                                         unit=_unit, trans_a=_trans_a, trans_b=_trans_b)
                else:
                    type_constraint = None

                params_all[_type][_index][_coeff]["free"] = not _constrained
                params_all[_type][_index][_coeff]["type_constrain"] = type_constraint

        return params_all

    # def generate_params_free(self):





    def _gen_mapping_params(self, additional_value: str = None):
        """
        can generate a mapping for all the coefficients
        :param args: (str) additionnal keys of the coefficient dictionnary to return
        :return:
        """


        params_all = self.params_all

        type_list = []
        index_list = []
        coeff_list = []
        additional_value_list = []

        for key0 in params_all.keys():
            for ii, elems in enumerate(params_all[key0]):
                for key1 in elems.keys():
                    type_list.append(key0)
                    index_list.append(ii)
                    coeff_list.append(key1)
                    if additional_value is not None:
                        additional_value_list.append(params_all[key0][ii][key1][additional_value])
        if additional_value is not None:
            return type_list, index_list, coeff_list, additional_value_list
        else:
            return type_list, index_list, coeff_list


    def _gen_mapping_params_from_unique_index(self, unique_index_to_find):

        type_list, index_list, coeff_list, unique_index = self._gen_mapping_params("index")
        ii = np.where(np.array(unique_index) == unique_index_to_find)[0][0]
        return type_list[ii], index_list[ii], coeff_list[ii]

    def _map_user_notation_in_yaml(self):
        type_list, index_list, coeff_list, index_in_fit_yaml_list = self._gen_mapping_params("index_in_fit_yaml")
        type_list, index_list, coeff_list, unique_index_list = self._gen_mapping_params("index")

        notation_user = self.parinfo["constrain_lines"]["coeff_notation"]
        notation_user_to_unique_index = {}
        for ii, elem1 in enumerate(notation_user):
            for jj, elem2 in enumerate(elem1):
                l = [[ii, jj] == n for n in index_in_fit_yaml_list]
                idx = np.where(l)[0][0]

                notation_user_to_unique_index[elem2] = unique_index_list[idx]
        return notation_user_to_unique_index

    def _build_self_params(self):
        raise NotImplementedError


    def _parse_bounds(self, bounds_to_parse: list[str | int | float],
                      unit: str, trans_a: float | int, trans_b: float | int):
        """
            Generates the bounds of self.params_all from the list taken out of the template.
            bounds can either have the following format:
            - bounds = [0, 1000] (for constants bounds)
            - bounds = {"ref": "guess", "operation": "plus", "value: 0.3} (value must have the Constant.ref unit)

            :param bounds_to_parse: the origin bounds from the yaml file.
            :param unit: the units of the origin bounds for the given coefficient
            :param trans_a: the trans_a from the yaml file for the given coefficient
            :param trans_b: the trans_b from the yaml file for the given coefficient
        """
        bounds_output = []

        for ii, bbb in enumerate(bounds_to_parse):
            if type(bbb) is str:
                # parse bound
                m = self.bound_equation_re.match(bbb)
                if m is None:
                    raise ValueError("bound string could not be parsed")
                d = m.groupdict()
                val = np.float64(d["value"])
                val = u.Quantity(val, unit) * trans_a + u.Quantity(trans_b, unit)
                val = self._transform_to_conventional_unit(val)
                bounds_output.append({
                    "ref": d["ref"],
                    "operation": d["operation"],
                    "value": np.float64(val.value),
                })
            elif (type(bbb) is float) or (type(bbb) is int):
                bbb_ =  np.float64(bbb)
                bbb_ = u.Quantity(bbb_, unit) * trans_a + u.Quantity(trans_b, unit)
                bbb_ = self._transform_to_conventional_unit(bbb_)

                bounds_output.append(np.float64(bbb_))
            else:
                raise TypeError("bounds must be a string or float or integer")
        return bounds_output

    def _parse_constrains_(self, constrain_to_parse: str,
                           unit: str, trans_a: float | int, trans_b: float | int
                           ):
        constrain_output = None
        m = self.constrain_equation_re.match(constrain_to_parse)
        if constrain_to_parse == "N/A":
            return constrain_output
        if m is None:
            raise ValueError("constrain_type string could not be parsed")
        d = m.groupdict()
        val = np.float64(d["value"])
        val = u.Quantity(val, unit) * trans_a + u.Quantity(trans_b, unit)
        val = self._transform_to_conventional_unit(val)

        map_yaml_notation = self._map_user_notation_in_yaml()
        constrain_output = {
            "ref": map_yaml_notation[d["ref"]],
            "operation": d["operation"],
            "value": val.value,
        }
        return constrain_output



    @staticmethod
    def _transform_to_conventional_unit(quantity: u.Quantity) -> u.Quantity:
        """
        transform an u.quantity into either a nm or a W/ (m2 sr nm), which are the conventional units for the fitting.
        :param quantity:
        """
        if quantity.unit.is_equivalent(Constants.conventional_lambda_units):
            return quantity.to(Constants.conventional_lambda_units)
        elif quantity.unit.is_equivalent(Constants.conventional_spectral_units):
            return quantity.to(Constants.conventional_spectral_units)
        else:
            raise ValueError(f"Cannot convert {quantity} to conventional unit")

    def _check_parinfo(self, parinfo=None):
        """
        Check if parinfo or self._parinfo has the valid format.
        :param parinfo: dictionnary containing information about the lines divided into the 'info' and 'fit' categories.
        :return: True if parinfo is valid. raise error otherwise.
        """
        if parinfo is not None:
            test = parinfo
        else:
            test = self.parinfo

        if type(test) is not dict:
            raise ValueError(f"Fittemplate must be a dictionary")
        if (
                ((len(test.keys()) != 3) and (len(test.keys()) != 4))
                | ("info" not in test.keys())
                | ("fit" not in test.keys())
                | ("main_line" not in test.keys())
        ):
            raise ValueError(f"Fittemplate must be a dictionary with 2 or 3 keys")
        keys = ["name",
                "elem",
                "ion",
                "wave",
                "unit_wave",
                "lvl_low",
                "lvl_up",
                "legend",]
        formats = [str,
                   str,
                   int,
                   float,
                   str,
                   str,
                   str,
                   str,]
        if str(type(test["info"])) != "<class 'list'>":
            raise ValueError(f"Fittemplate['info'] must be a list with lines information")
        for elem in test["info"]:
            for key, format_ in zip(keys, formats):
                if key not in elem.keys():
                    raise ValueError(f"key {key} must exist")
                elif type(elem[key]) is not format_:
                    raise ValueError(f"key {key} must be in {format_}")

            if type(test["fit"]) is not dict:
                raise ValueError(f"Fittemplate['fit'] must be a dictionary")

        keys = [
            "type",
            "name",
            "n_components",
            "guess",
            "coeff_type",
            "units",
            "max_arr",
            "min_arr",
            "trans_a",
            "trans_b",
        ]
        formats = [
            [list, str],
            [list, str],
            [list, int],
            [list, list, float | int],
            [list, list, str],
            [list, list, str],
            [list, list, float | int | str],
            [list, list, float | int | str],
            [list, list, float | int],
            [list, list, float | int],
        ]
        self._check_format(formats, keys, test["fit"])

        for ii in range(len(test["fit"]["n_components"])):
            for elem in keys:
                if elem not in ["name", "n_components", "type"]:
                    if len(test["fit"][elem][ii]) != test["fit"]["n_components"][ii]:
                        raise ValueError("Inconsistent lengths between fit parameters in Fittemplate")

        if (len(test.keys()) == 4) and ("constrain_lines" not in test.keys()):
            raise ValueError(f"Fittemplate third key must be 'constrained_lines'")

        if len(test.keys()) == 4:
            keys = [
                "constrained",
                "type_constrain",
                "coeff_notation",
            ]
            formats = [
            [list, list, bool],
            [list, list, str],
            [list, list, str],
            ]
            self._check_format(formats, keys, test["constrain_lines"])


        name_fit = test["fit"]["name"]
        name_info = [n["name"] for n in test["info"]]

        if len(set(name_fit)) != len(name_fit):
            breakpoint()
            raise ValueError(f"duplicated names in the fit list name !")

        if len(set(name_info)) != len(name_info):
            raise ValueError(f"duplicated names in the info list name !")

        for n in name_fit:
            if (n in name_info) and (n != "background"):
                ValueError(f"a line in the fit list name is not present in the info list !")

        for n in name_info:
            if n in name_fit:
                ValueError(f"line in the info list is not present in the fit name list !")
        return True

    def _check_format(self, formats, keys, dict_):
        for key, format_ in zip(keys, formats):
            if key not in dict_.keys():
                raise ValueError(f"key {key} must exist ")

            if not isinstance(dict_[key], format_[0]):
                breakpoint()
                raise ValueError(f"key {key}(0) must have the right format {format_[0]}")
            for xx in range(len(dict_[key])):
                if not isinstance(dict_[key][xx], format_[1]):
                    breakpoint()
                    raise ValueError(f"key {key}(1) must have the right format {format_[1]}")
                if len(format_) == 3:
                    for yy in range(len(dict_[key][xx])):
                        if not isinstance(dict_[key][xx][yy], format_[2]):
                            breakpoint()
                            raise ValueError(f"key {key}(2) must have the right format {format_[2]}")

    # def __repr__(self):
    #     pass

