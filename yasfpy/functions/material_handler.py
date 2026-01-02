"""Material table helpers.

Contains utilities for reading/interpolating refractive index tables used by YASF.
"""

import hashlib
import io
import os
import re
from importlib.resources import files

# import urllib.request
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests as req
import yaml

HASH_TYPE = "sha256"


def material_handler(links, cache: bool = True, local: bool = True):
    """
    Handles the processing of material data from various sources.

    Args:
        links (str or list): The link(s) to the material data source(s).
        cache (bool): cache the downloaded file using sha256 string
        local (bool): load locally cached files if available

    Returns:
        (dict): A dictionary containing the processed material data and information.

    """
    if not isinstance(links, list):
        links = [links]

    data_path = Path(f"{files(__package__) / 'data'}")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data = dict(ref_idx=pd.DataFrame(columns=["wavelength", "n", "k"]), material=None)
    for link in links:
        link = link.strip()
        if link[:4] == "http":
            is_cached = False
            h = hashlib.new(HASH_TYPE)
            h.update(link.encode("utf-8"))
            cache_file_name = os.path.join(data_path, f"{h.hexdigest()}.csv")
            material_file_name = os.path.join(data_path, f"{h.hexdigest()}.txt")
            if os.path.isfile(cache_file_name):
                df = pd.read_csv(cache_file_name)
                material = "NaN"
                with open(material_file_name, "r") as fh:
                    material = fh.read()
                is_cached = True
                # print(f"Reading {link} from cache: {cache_file_name}.")
            elif "refractiveindex.info" in link:
                df, material = handle_refractiveindex_info(link)
                # data["ref_idx"] = pd.concat([data["ref_idx"], df])
                # data["material"] = material
            elif "eodg.atm.ox.ac.uk" in link:
                df, material = handle_eodg(link)
                # data["ref_idx"] = pd.concat([data["ref_idx"], df])
                # data["material"] = material
            else:
                print(f"No matching handler found for {link}")
                continue
            data["ref_idx"] = pd.concat([data["ref_idx"], df])
            data["material"] = material

            if not is_cached:
                # print(f"{link} was not cached. Not caching at {cache_file_name}.")
                df.to_csv(cache_file_name, index=False)
                with open(material_file_name, "w") as fh:
                    fh.write(material)
        else:
            if ".csv" in link:
                df, material = handle_csv(link)
                data["ref_idx"] = pd.concat([data["ref_idx"], df])
                data["material"] = material
            else:
                print("No matching handler found for file type")
    data["ref_idx"] = data["ref_idx"].sort_values(by=["wavelength"])
    return data


def handle_refractiveindex_info(url):
    """
    Retrieves refractive index data from a given URL and processes it.

    Args:
        url (str): The URL to retrieve the refractive index data from.

    Returns:
        (tuple): A tuple containing the processed data as a pandas DataFrame and the material name.

    Raises:
        Exception: If the data retrieval fails.

    """
    url_split = url.replace("=", "/").split("/")
    material = unquote(url_split[-2])

    if np.any([("data_csv" in part) or ("data_txt" in part) for part in url_split]):
        print("Please use the [Full database record] option for refractiveindex.info!")
        print("Reverting url:")
        print(f" from: {url}")
        url_split[3] = "database"
        url = "/".join(url_split)
        print(f" to:   {url}")

    # req = urllib.request.Request(url)
    # with urllib.request.urlopen(req) as resp:
    resp = req.get(url)
    if resp.status_code >= 400:
        raise Exception(f"Failed to retrieve data from {url}")
    # data = resp.read()
    # data = data.decode("utf-8")
    data = resp.text
    data_yml = yaml.safe_load(data)
    header_yml = ["wavelength", "n", "k"]
    data = pd.DataFrame(columns=["wavelength", "n", "k"])
    for line in data_yml["DATA"]:
        df = None
        if "tabulated" in line["type"].lower():
            # elif line['type'].lower()[-2:] == ' n':
            #   header_yml=['wavelength', 'n']
            if line["type"].lower()[-2:] == " k":
                header_yml = ["wavelength", "k", "n"]
            df = pd.read_csv(
                io.StringIO(line["data"]),
                delim_whitespace=True,
                header=None,
                names=header_yml,
            )
        elif "formula" in line["type"].lower():
            if line["type"].lower() == "formula 1":
                wavelengths = [float(c) for c in line["wavelength_range"].split()]
                wavelengths = np.arange(wavelengths[0], wavelengths[1], 0.1)
                coefficients = np.array(
                    [float(c) for c in line["coefficients"].split()]
                )
                ref_idx = lambda x: np.sqrt(
                    1
                    + np.sum(
                        [
                            coefficients[i] * x**2 / (x**2 - coefficients[i + 1] ** 2)
                            for i in range(1, len(coefficients), 2)
                        ],
                        axis=0,
                    )
                )
                df = pd.DataFrame(columns=["wavelength", "n", "k"])
                df["wavelength"] = wavelengths
                df["n"] = ref_idx(wavelengths)

        if df is not None:
            df = df.fillna(0)
            data = pd.concat([data, df])

    return data, material


def handle_eodg(url):
    """
    Handles the EODG material data retrieval from a given URL.

    Args:
        url (str): The URL of the EODG material data.

    Returns:
        (tuple): A tuple containing the retrieved material data and the material name.

    Raises:
        Exception: If the data retrieval from the URL fails.
    """
    url_split = url.split("/")
    material = unquote(url_split[6])

    resp = req.get(url)
    if resp.status_code >= 400:
        raise Exception(f"Failed to retrieve data from {url}")
    data = resp.text
    data_format = [
        s.lower() for s in re.search(r"#FORMAT=(.*)\n", data).group(1).split()
    ]
    header_yml = ["wavelength", "n", "k"]
    if "n" not in data_format:
        header_yml = ["wavelength", "k", "n"]

    data = re.sub(r"^#.*\n", "", data, flags=re.MULTILINE)
    data = pd.read_csv(
        io.StringIO(data), delim_whitespace=True, header=None, names=header_yml
    )
    data = data.fillna(0)
    if "wavn" in data_format:
        data["wavelength"] = 1e4 / data["wavelength"]
        data = data.iloc[::-1]

    return data, material


def handle_csv(path):
    """
    Read a CSV file containing material data and extract the material name and data.

    Args:
        path (str): The path to the CSV file.

    Returns:
        (tuple): A tuple containing the material data as a pandas DataFrame and the material name.
    """
    name = re.split(r"\._-", path)
    material = unquote(name[0])
    data = pd.read_csv(
        path, delim_whitespace=False, header=0, names=["wavelength", "n", "k"]
    )
    return data, material
