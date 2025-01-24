from ..FittingModel import FittingModel
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt



def test_init_fit_model():
    path_to_template_complex = Path("./core/test/fit_templates/ne_8_770_42_1c.template_complex.yaml")
    path_to_template = Path("./core/test/fit_templates/ne_8_770_42_1c.template.yaml")

    # f2 = FittingModel(filename=str(path_to_template_complex.absolute()))
    # print(f2.params_all)
    # print(f2.params_free["notation"])
    #
    # assert f2.params_free["notation"] == ['I0', 'x0', 's0']
    #
    # print(f2._str_fitting_function)
    # print(f2._str_jacobian_function)

    #
    f1 = FittingModel(filename=str(path_to_template.absolute()), use_jit=True, cache=True)
    print(f1.params_all)
    print(f1.params_free["notation"])
    print(f1.fitting_function)
    print(f1.jacobian_function)

    x = np.linspace(0, 10, 1000)
    y = f1.fitting_function(x, I0 = 5,x0 = 5, s0 = 1,a0 = 0.2)
    plt.plot(x, y)
    plt.show()


