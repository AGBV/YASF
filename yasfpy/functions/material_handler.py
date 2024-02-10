import re
import io
import pandas as pd
import yaml
# import urllib.request
from urllib.parse import unquote
import requests as req

import numpy as np

def material_handler(links):
    if not isinstance(links, list):
        links = [links]

    data = dict(ref_idx=pd.DataFrame(columns=["wavelength", "n", "k"]), material=None)
    for link in links:
        link = link.strip()
        if link[:4] == "http":
            if "refractiveindex.info" in link:
                df, material = handle_refractiveindex_info(link)
                data["ref_idx"] = pd.concat([data["ref_idx"], df])
                data["material"] = material
            elif "http://eodg.atm.ox.ac.uk" in link:
                df, material = handle_eodg(link)
                data["ref_idx"] = pd.concat([data["ref_idx"], df])
                data["material"] = material
            else:
                print("No mathing handler found for url")
        else:
            if ".csv" in link:
                df, material = handle_csv(link)
                data["ref_idx"] = pd.concat([data["ref_idx"], df])
                data["material"] = material
            else:
                print("No mathing handler found for file type")
    # data['ref_idx'] = data['ref_idx'].sort_values(by=['wavelength'])
    return data


def handle_refractiveindex_info(url):
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
    #with urllib.request.urlopen(req) as resp:
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
                            coefficients[i]
                            * x**2
                            / (x**2 - coefficients[i + 1] ** 2)
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
    url_split = url.split("/")
    # material = unquote(url_split[-1][:-3]).replace('_', ' ')
    material = unquote(url_split[6])

    # req = urllib.request.Request(url)
    # with urllib.request.urlopen(req) as resp:
    # data = resp.read()
    # data = data.decode("iso-8859-1")
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
    # eodg uses wavenumbers in cm-1 instead of wavelengths in um, hence um = 1e4 / cm-1
    if "wavn" in data_format:
        data["wavelength"] = 1e4 / data["wavelength"]
        data = data.iloc[::-1]

    return data, material


def handle_csv(path):
    name = re.split(r"\._-", path)
    material = unquote(name[0])
    data = pd.read_csv(
        path, delim_whitespace=False, header=0, names=["wavelength", "n", "k"]
    )
    return data, material
