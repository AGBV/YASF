import unittest
import glob
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from yasfpy.functions.material_handler import material_handler

class TestHandlers(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cls.data = {}
    #     cls.path = f"tests/data/celes_*.fits"
    #     cls.id_regex = r"celes_(.+)\.fits"
    #     cls.relative_precision = 2e-3

    def test_refractiveindex_info_fe_querry(self):
        url = "https://refractiveindex.info/database/data-nk/main/Fe/Querry.yml"
        data = material_handler(url)
        print(data)
