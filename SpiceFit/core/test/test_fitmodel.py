from ..FittingModel import FittingModel
from pathlib import Path


def test_init_fit_model():
    path_to_template_complex = Path("./core/test/fit_templates/ne_8_770_42_1c.template_complex.yaml")
    path_to_template = Path("./core/test/fit_templates/ne_8_770_42_1c.template.yaml")

    f2 = FittingModel(filename=str(path_to_template_complex.absolute()))
    print(f2.params_all)
    print(f2.params_free["notation"])

    assert f2.params_free["notation"] == ['I0', 'x0', 's0']

    print(f2.str_fitting_function_)
    #
    # f1 = FittingModel(filename=str(path_to_template.absolute()))
    # print(f1.params_all)
    # print(f1.params_free["notation"])
    # print(f1.str_fitting_function_)

