import unittest

# import glob
# import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd
from itertools import takewhile

from yasfpy.functions.material_handler import material_handler


class TestHandlers(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cls.data = {}
    #     cls.path = f"tests/data/celes_*.fits"
    #     cls.id_regex = r"celes_(.+)\.fits"
    #     cls.relative_precision = 2e-3

    def test_refractiveindex_info_fe_querry(self):
        csv_file = "tests/data/fe_querry.csv"
        csv_data = pd.read_csv(csv_file, sep=r"\s+", comment="#")

        header = None
        with open(csv_file, "r") as fobj:
            # takewhile returns an iterator over all the lines
            # that start with the comment string
            header = takewhile(lambda s: s.startswith("#"), fobj)
            header = [s.strip("# \n").split(": ") for s in header]
            header = {s[0]: s[1].strip() for s in header}
        self.assertTrue(header is not None)

        url = header["data"]
        url_data = material_handler(url)

        pd.testing.assert_frame_equal(csv_data, url_data["ref_idx"])
