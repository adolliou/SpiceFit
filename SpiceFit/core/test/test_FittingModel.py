import os
from glob import glob
from ..FittingModel import FittingModel
from pathlib import Path


class TestFittingModel():

    def test_constructor(self):
        basis_yaml_templates_file = os.path.join(Path(__file__).parents[2],
         f"TemplatesLines/*.template.yaml")

        file_list      = glob(basis_yaml_templates_file)

        for file in file_list:
            model = FittingModel(file)
