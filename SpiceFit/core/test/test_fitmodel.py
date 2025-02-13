from ..FittingModel import FittingModel
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt



def test_init_fit_model():
    path_to_template_complex = Path("./core/test/fit_templates/ne_8_770_42_1c.template_complex.yaml")
    path_to_template = Path("./core/test/fit_templates/ne_8_770_42_1c.template.yaml")

    #
    f1 = FittingModel(filename=str(path_to_template.absolute()), use_jit=True, cache=True)


