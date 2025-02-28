from ..FittingModel import FittingModel
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os


def test_init_fit_model():
    path_to_template_complex = os.path.join(Path(__file__).parents[0],"fit_templates/ne_8_770_42_1c.template_complex.yaml")
    path_to_template = os.path.join(Path(__file__).parents[0],"fit_templates/ne_8_770_42_1c.template.yaml")

    #
    f1 = FittingModel(filename=path_to_template, use_jit=True, cache=True)




