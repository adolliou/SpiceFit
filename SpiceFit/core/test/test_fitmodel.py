from ..FittingModel import FittingModel
from pathlib import Path


def test_init_fitmodel():
    path_to_template = Path("./core/test/fit_templates/ne_8_770_42_1c.template.yaml")
    f = FittingModel(filename=str(path_to_template.absolute()))
    print(f.params_all)