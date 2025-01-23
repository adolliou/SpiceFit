import numpy as  np
import re
import copy


class TestClass:

    bound_equation_re = re.compile(
        r'''
    (?P<ref>(guess))
    _(?P<operation>(plus|minus|times))
    _(?P<value>\d*\.\d*)
    ''',
        re.VERBOSE)



    def __init__(self):



        self._params_all = {
            "guess": 300,
            "bounds": ["guess_plus_03.06", 350]

        }

        self._params_free = {
            "guess": 300,
            "bounds": [200, 300]
        }

    @property
    def params_all(self):
        return copy.deepcopy(self._params_all)

    @params_all.setter
    def params_all(self, params_all: dict):

        self._params_all = params_all
        for ii, bbb in enumerate(self.params_all["bounds"]):

            if str(type(bbb)) == "<class 'str'>":
                # parse bound
                m = self.bound_equation_re.match(bbb)
                if m is None:
                    raise ValueError("bound string could not be parsed")
                d = m.groupdict()
                if d["ref"] == "guess":
                    value = float(d["value"])
                    if d["operation"] == "plus":

                        self.params_free["bounds"][ii] = self.params_all["guess"] + value
                    elif d["operation"] == "minus":
                        self.params_free["bounds"][ii] = self.params_all["guess"] - value
                    elif d["operation"] == "times":
                        self.params_free["bounds"][ii] = self.params_all["guess"] * value
                else:
                    raise NotImplementedError("unknown guess")


            else:
                self.params_free["bounds"][ii] = self._params_all["bounds"][ii]


    @property
    def params_free(self):
        return self._params_free

    @params_free.setter
    def params_free(self, params_free):
        self._params_free = params_free


if __name__ == '__main__':
    test = TestClass()

    print(test.params_free["bounds"])
    print(test.params_all["bounds"])
    print("\n====================\n")
    d = copy.deepcopy(test.params_all)
    d["guess"] = 400
    test.params_all = d

    print(test.params_free["bounds"])
    print(test.params_all["bounds"])
    print("\n====================\n")

    test.params_all["guess"] = 300

    print(test.params_free["bounds"])
    print(test.params_all["bounds"])

    constrain_equation_re = re.compile(
        r'''
    (?P<ref>(.+?))
    _(?P<operation>(plus|minus|times))
    _(?P<value>\d*\.\d*)
    ''',
        re.VERBOSE)

    A = "x0_minus_66.35"
    m = constrain_equation_re.match(A)
    d = m.groupdict()

    # constrain_equation_re = re.compile(
    #     r'''
    # (?P<operation>(plus|minus|times))
    # _(?P<value>\d*\.\d*)
    # ''',
    #     re.VERBOSE)
    #
    # A = "minus_66.35"
    # m = constrain_equation_re.match(A)


