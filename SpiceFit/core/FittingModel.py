import warnings

import astropy.units as u
import numpy as np
from yaml import safe_load
from ..util.constants import Constants
import copy
import re
import string
from pathlib import Path
import os
import importlib.util
import sys
import secrets


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

    # Template used to compute fitting and Jacobian functions

    def __init__(self, filename: str = None, parinfo: dict = None, verbose: int = 1,
                 use_jit: bool = False, cache: bool = True) -> None:
        """

        :param filename: (optional)
        :param parinfo: dictionary, should have the same keys as a parinfo dictionary
        """
        self.verbose = verbose
        self._parinfo = {"fit": {}, "info": {}}
        self.cache = cache
        self.use_jit = use_jit
        jit_str = ""
        if self.use_jit:
            jit_str = f"\n@ jit(nopython=True, inline='always', cache={cache})"
        self.template_function = (""
                                  "\n{0}"
                                  f"{jit_str}"
                                  "\ndef {1}(x, {2}):"
                                  "\n\tm = {3}"
                                  # "\n\tx=np.array(x, dtype=np.float64)"
                                  "\n\tsum = np.zeros(len(x), dtype=np.float64)"
                                  "\n\t"
                                  "")

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
        self.generate_free_params()

        # self._str_fitting_function = self.create_fitting_function_str()
        # self._str_jacobian_function = self.   create_jacobian_function_str()

        self.fitting_function = None
        self.jacobian_function = None

        self.generate_callable_fitting_jacobian_function_from_module()
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
    def fitting_function(self):
        """
        self.params_all getter.
        """
        return self._fitting_function

    @fitting_function.setter
    def fitting_function(self, fitting_function: callable):

        self._fitting_function = fitting_function

    @property
    def jacobian_function(self):
        """
        self.params_all getter.
        """
        return self._jacobian_function

    @jacobian_function.setter
    def jacobian_function(self, jacobian_function: callable):

        self._jacobian_function = jacobian_function

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

    def generate_free_params(self):
        self._params_free = {
            "guess": [],
            "bounds": [[], []],
            "unit": [],
            "index": [],
            "notation": [],
        }
        type_list, index_list, coeff_list = self.gen_mapping_params()
        for type_, idx_, coeff_ in zip(type_list, index_list, coeff_list):

            if self.params_all[type_][idx_][coeff_]["free"]:
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

                self._params_free["guess"].append(self.params_all[type_][idx_][coeff_]["guess"], )
                self._params_free["bounds"][0].append(bounds[0])
                self._params_free["bounds"][1].append(bounds[1])

                self._params_free["unit"].append(self.params_all[type_][idx_][coeff_]["unit"], )
                self._params_free["index"].append(self.params_all[type_][idx_][coeff_]["index"], )
                self._params_free["notation"].append(self.params_all[type_][idx_][coeff_]["notation"], )
        self._params_free["guess"] = np.array(self._params_free["guess"], dtype=np.float64)
        self._params_free["bounds"] = tuple(self._params_free["bounds"])

    def create_fitting_function_str(self):
        s = self.template_function.format(self._function_generate_imports_str(),
                                          'fitting_function',
                                          ','.join(self.params_free["notation"]),
                                          0,
                                          )
        s += '\n\t'
        for function_type in self.params_all.keys():
            for ii in range(len(self.params_all[function_type])):
                if function_type == "polynomial":
                    s += self._polynom_to_fitting_str(index_polynom=ii)
                elif function_type == "gaussian":
                    s += self._gauss_to_fitting_str(index_gaussian=ii)
                else:
                    raise NotImplementedError
        s += "\n\n\treturn sum"
        return s

    def create_jacobian_function_str(self):
        """
        creates a str of the code for the Jacobian function.
        :return:
        """
        _type_list, _index_list, _coeff_list = self.gen_mapping_params()
        for _type, _index, _coeff in zip(_type_list, _index_list, _coeff_list):
            if not self.params_all[_type][_index][_coeff]["free"]:
                if self.verbose > 1:
                    warnings.warn("The jacobian function is set to None due to Constrained parameters ")
                return None

        s = self.template_function.format("",
                                          'jacobian_function',
                                          ','.join(self.params_free["notation"]),
                                          f'np.zeros((len(x), {len(self.params_free["guess"])}), dtype=np.float64)',
                                          )

        s += '\n\t'
        _type_list, _index_list, _coeff_list = self.gen_mapping_params()

        # jacobian_matrix_s = np.array((1, len(_coeff_list)), dtype=str)
        index = 0
        for function_type in self.params_all.keys():

            for ii in range(len(self.params_all[function_type])):
                ncoeffs = 0
                if function_type == "polynomial":
                    jac_str = self._polynom_to_jacobian_str(index_polynom=ii)
                    ncoeffs = len(jac_str)
                    for jj in range(ncoeffs):
                        s += "\n\tm[:, {0}] = {1}".format(int(index + jj), jac_str[jj])
                elif function_type == "gaussian":
                    jac_str = self._gauss_to_jacobian_str(index_gaussian=ii)
                    ncoeffs = len(jac_str)
                    for jj in range(ncoeffs):
                        s += "\n\tm[:, {0}] = {1}".format(int(index + jj), jac_str[jj])
                else:
                    raise NotImplementedError

                index += ncoeffs

        s += "\n\n\treturn m"
        return s

    # def generate_callable_fitting_jacobian_function_exec(self):

        # exec(self.create_fitting_function_str())
        # self.fitting_function = fitting_function
        # self.jacobian_function = jacobian_function

    def generate_callable_fitting_jacobian_function_from_module(self, directory_path: str = None, filename: str = None):
        """

        :param directory_path:
        :param filename:
        """
        if directory_path is None:
            directory_path = "./.fitting_jacobian_functions"

        if filename is None:
            filename = "fitting_jacobian_functions.py"

        Path(directory_path).mkdir(exist_ok=True)
        with open(os.path.join(directory_path, filename), "w") as f:
            f.write(self.create_fitting_function_str())
            if self.create_jacobian_function_str() is not None:
                f.write("\n\n\n")
                f.write(self.create_jacobian_function_str())

        sys.path.append(directory_path)
        import fitting_jacobian_functions
        # spec = importlib.util.spec_from_file_location(location=os.path.join(directory_path, filename),
        #                                               name=filename, )
        # module = importlib.util.module_from_spec(spec)
        # sys.modules[filename] = module
        # spec.loader.exec_module(module)

        # module = self._load_module(source=os.path.join(directory_path, filename), module_name = "FittingJacobian")
        # self.fitting_function = getattr(module, "fitting_function")
        self.fitting_function = fitting_jacobian_functions.fitting_function

        if self.create_jacobian_function_str() is not None:
            # self.jacobian_function = getattr(module, "jacobian_function")
            self.jacobian_function = fitting_jacobian_functions.jacobian_function
        else:
            self.jacobian_function = None
        # del sys.modules["FittingJacobian"]

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

        type_list, index_list, coeff_list = self.gen_mapping_params()
        for type_, idx_, coeff_ in zip(type_list, index_list, coeff_list):

            for key in list(_params_all[type_][idx_][coeff_].keys()):
                del _params_all[type_][idx_][coeff_][key]

            _params_all[type_][idx_][coeff_][key] = {
                "guess": None,
                "bounds": None,
                "unit": None,
                "index": None
            }

            if not self.params_all[type_][idx_][coeff_]["free"]:
                del _params_all[type_][idx_][coeff_][key]
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
        index_polynomial = 0

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

                for idx in [idxi, idxx, idxs]:
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
                _name = name_list[ii][idx]
                _guess = (u.Quantity(_guess, _unit) * _trans_a + u.Quantity(_trans_b, _unit))
                guess = self._transform_to_conventional_unit(_guess)
                _bounds_to_parse = [_min_arr, _max_arr]
                bounds = self._parse_bounds(bounds_to_parse=_bounds_to_parse,
                                            unit=_unit, trans_a=_trans_a, trans_b=_trans_b)

                notation = f'{str(coeff)}{index:d}'

                params_all[type][-1][coeff] = {
                    "guess": guess.value,
                    "bounds": bounds,
                    "unit": str(guess.unit),
                    "notation": notation,
                    "index": index,
                    "free": True,
                    "type_constrain": None,
                    "name_line": _name,

                    "index_in_fit_yaml": [ii, idx]
                }

        self.params_all = params_all
        if "constrain_lines" in self.parinfo.keys():
            _type_list, _index_list, _coeff_list = self.gen_mapping_params()

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

    def gen_mapping_params(self, additional_value: str = None):
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

    def gen_coeff_from_unique_index(self, unique_index_to_find):

        type_list, index_list, coeff_list, unique_index = self.gen_mapping_params("index")
        ii = np.where(np.array(unique_index) == unique_index_to_find)[0][0]
        return self.params_all[type_list[ii]][index_list[ii]][coeff_list[ii]]

    def _map_user_notation_in_yaml(self):
        type_list, index_list, coeff_list, index_in_fit_yaml_list = self.gen_mapping_params("index_in_fit_yaml")
        type_list, index_list, coeff_list, unique_index_list = self.gen_mapping_params("index")

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
                bbb_ = np.float64(bbb)
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

    @staticmethod
    def _function_generate_imports_str():
        return ("import numpy as np\n"
                "from numba import jit\n\n\n")

    def _polynom_to_fitting_str(self, index_polynom: int):
        """
        returns the str to add to the fitting function str for the given (index_polynom) gaussian function
        :param index_polynom:
        :return:
        """
        s = "\n\tsum = sum + "
        if "polynomial" not in self.params_all:
            return ""
        coeffs = self.params_all["polynomial"][index_polynom]

        letters = list(string.ascii_lowercase)
        npoly = len(coeffs) - 1

        for ii in range(len(coeffs)):
            cstr = self._generate_str_fitting_function_for_params(coeffs[letters[ii]])
            s += f'{cstr} * x ** ({npoly - ii})'
        return s

    def _gauss_to_fitting_str(self, index_gaussian: int):
        """
        returns the str to add to the fitting function str for the given (index_gaussian) gaussian method
        gaussian(x) = I*exp(((x - mu)**2)/(2 * sigma**2))
        :param index_gaussian: index of the gaussian function to write
        """

        s = "\n\tsum = sum + "
        if "gaussian" not in self.params_all:
            return ""
        coeffs = self.params_all["gaussian"][index_gaussian]
        i_str = self._generate_str_fitting_function_for_params(coeffs["I"])
        x_str = self._generate_str_fitting_function_for_params(coeffs["x"])
        s_str = self._generate_str_fitting_function_for_params(coeffs["s"])

        s += f"{i_str} * np.exp( - ( ( {x_str} - x )**2 ) / (2 * ( {s_str} )**2 ))\n"
        return s

    def _polynom_to_jacobian_str(self, index_polynom: int):
        """
        creates a matrix of str corresponding to the Jacobian of the given polynomial function.
        each matrix index must be to the corresponding indexes of the global Jacobian matrix to provide
        the complete jacobian

        coeff i
        f = an * x**n + ... + ai * x**i + .... + a0 * x**0
        J[0, i] = df/dai = x**i

        :param index_polynom:
        :return:
        """
        # s = "\n\tsum+="

        if "polynomial" not in self.params_all:
            return ""
        coeffs = self.params_all["polynomial"][index_polynom]
        npoly = len(coeffs) - 1

        jac_matrix_str = []
        for ii in range(len(coeffs)):
            jac_matrix_str.append(f"x**{npoly - ii:d}")
        return jac_matrix_str

    def _gauss_to_jacobian_str(self, index_gaussian: int):
        """
        returns the str to add to the jacobian function str for the given (index_gaussian) gaussian method
        f(x) = I*exp( - ((x - mu)**2)/(2 * sigma**2))
        df/dI =                                                        exp(- ((x - mu)**2)/(2 * sigma**2))
        df/dmu =        I                 * (x - mu)/(sigma**2)      * exp(- ((x - mu)**2)/(2 * sigma**2))
        df/dsigma =     I * sigma         * (x - mu)**2/(sigma**3)   * exp(- ((x - mu)**2)/(2 * sigma**2))

        :param index_gaussian: index of the gaussian function to write
        """

        if "gaussian" not in self.params_all:
            return ""
        coeffs = self.params_all["gaussian"][index_gaussian]
        i_str = self._generate_str_fitting_function_for_params(coeffs["I"])
        x_str = self._generate_str_fitting_function_for_params(coeffs["x"])
        s_str = self._generate_str_fitting_function_for_params(coeffs["s"])
        jac_matrix_str = []

        exp_str = "{0} * np.exp(- ((x - {1})**2)/(2 * {2}**2)) "
        # df/dI
        jac_matrix_str.append(exp_str.format("1", x_str, s_str))
        # df/dx
        jac_matrix_str.append(" * ".join(["((x - {0})/({1}**2))".format(x_str, s_str),
                                          exp_str.format(i_str, x_str, s_str)]))
        # df/ds
        jac_matrix_str.append(" * ".join(
            ["{0}".format(s_str),
             "(((x - {0})**2)/({1}**3))".format(x_str, s_str),
             exp_str.format(i_str, x_str, s_str)
             ]
        ))
        return jac_matrix_str

    def _generate_str_fitting_function_for_params(self, a: dict):
        s = ""
        if a["free"]:
            s = a["notation"]
        else:
            dict_const = a["type_constrain"]
            s += "("
            b = self.gen_coeff_from_unique_index(dict_const["ref"])
            s += b["notation"]
            if dict_const["operation"] == "plus":
                s += " + "
            elif dict_const["operation"] == "minus":
                s += " - "
            elif dict_const["operation"] == "times":
                s += " * "
            s += str(dict_const["value"])
            s += ")"
        return s

    @staticmethod
    def _generate_str_jacobian_function_for_params(a: dict):
        s = ""
        if a["free"]:
            s = a["notation"]
        else:
            dict_const = a["type_constrain"]
            s += "("
            s += dict_const["ref"]
            if dict_const["operation"] == "plus":
                s += " + "
            elif dict_const["operation"] == "minus":
                s += " - "
            elif dict_const["operation"] == "times":
                s += " * "
            s += ")"
        return s

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
                "legend", ]
        formats = [str,
                   str,
                   int,
                   float,
                   str,
                   str,
                   str,
                   str, ]
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
                raise ValueError(f"key {key}(0) must have the right format {format_[0]}")
            for xx in range(len(dict_[key])):
                if not isinstance(dict_[key][xx], format_[1]):
                    raise ValueError(f"key {key}(1) must have the right format {format_[1]}")
                if len(format_) == 3:
                    for yy in range(len(dict_[key][xx])):
                        if not isinstance(dict_[key][xx][yy], format_[2]):
                            raise ValueError(f"key {key}(2) must have the right format {format_[2]}")

# from https://medium.com/@david.bonn.2010/dynamic-loading-of-python-code-2617c04e5f3f
#     @staticmethod
#     def _gensym(length=32, prefix="gensym_"):
#         """
#         generates a fairly unique symbol, used to make a module name,
#         used as a helper function for load_module
#
#         :return: generated symbol
#         """
#         alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
#         symbol = "".join([secrets.choice(alphabet) for i in range(length)])
#
#         return prefix + symbol
#
#     def _load_module(self,source, module_name=None):
#         """
#         reads file source and loads it as a module
#
#         :param source: file to load
#         :param module_name: name of module to register in sys.modules
#         :return: loaded module
#         """
#
#         if module_name is None:
#             module_name = self.gensym()
#
#
#
#         return module
# def __repr__(self):
#     pass
