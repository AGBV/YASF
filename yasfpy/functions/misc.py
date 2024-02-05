import re
import numpy as np

from scipy.special import spherical_jn
from scipy.special import hankel1, lpmv

from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon


def jmult_max(num_part, lmax):
    return 2 * num_part * lmax * (lmax + 2)


def multi2single_index(j_s, tau, l, m, lmax):
    return (
        j_s * 2 * lmax * (lmax + 2)
        + (tau - 1) * lmax * (lmax + 2)
        + (l - 1) * (l + 1)
        + m
        + l
    )


def single_index2multi(idx, lmax):
    j_s = idx // (2 * lmax * (lmax + 2))
    idx_new = idx % (2 * lmax * (lmax + 2))
    tau = idx_new // (lmax * (lmax + 2)) + 1
    idx_new = idx_new % (lmax * (lmax + 2))
    l = np.floor(np.sqrt(idx_new + 1))
    m = idx_new - (l * l + l - 1)
    return j_s, tau, l, m


def transformation_coefficients(pilm, taulm, tau, l, m, pol, dagger: bool = False):
    ifac = 1j
    if dagger:
        ifac *= -1

    # Polarized light
    if np.any(np.equal(pol, [1, 2])):
        if tau == pol:
            spher_fun = taulm[l, np.abs(m)]
        else:
            spher_fun = m * pilm[l, np.abs(m)]

        return (
            -1
            / np.power(ifac, l + 1)
            / np.sqrt(2 * l * (l + 1))
            * (ifac * (pol == 1) + (pol == 2))
            * spher_fun
        )

    # Unpolarized light
    return (
        -1
        / np.power(ifac, l + 1)
        / np.sqrt(2 * l * (l + 1))
        * (ifac + 1)
        * (taulm[l, np.abs(m)] + m * pilm[l, np.abs(m)])
        / 2
    )


def mutual_lookup(
    lmax: int,
    positions_1: np.ndarray,
    positions_2: np.ndarray,
    refractive_index: np.ndarray,
    derivatives: bool = False,
    parallel: bool = False,
):
    differences = positions_1[:, np.newaxis, :] - positions_2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    distances = distances[:, :, np.newaxis]
    e_r = np.divide(
        differences, distances, out=np.zeros_like(differences), where=distances != 0
    )
    cosine_theta = e_r[:, :, 2]
    # cosine_theta = np.divide(
    #   differences[:, :, 2],
    #   distances,
    #   out = np.zeros_like(distances),
    #   where = distances != 0
    # )
    # correct possible rounding errors
    cosine_theta[cosine_theta < -1] = -1
    cosine_theta[cosine_theta > 1] = 1
    sine_theta = np.sqrt(1 - cosine_theta**2)
    phi = np.arctan2(differences[:, :, 1], differences[:, :, 0])
    e_theta = np.stack(
        [cosine_theta * np.cos(phi), cosine_theta * np.sin(phi), -sine_theta], axis=-1
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)

    size_parameter = distances * refractive_index[np.newaxis, np.newaxis, :]

    if parallel:
        from src.functions.cpu_numba import compute_lookup_tables

        spherical_bessel, spherical_hankel, e_j_dm_phi, p_lm = compute_lookup_tables(
            lmax, size_parameter, phi, cosine_theta
        )
    else:
        p_range = np.arange(2 * lmax + 1)
        p_range = p_range[:, np.newaxis, np.newaxis, np.newaxis]
        size_parameter_extended = size_parameter[np.newaxis, :, :, :]
        spherical_hankel = np.sqrt(
            np.divide(
                np.pi / 2,
                size_parameter_extended,
                out=np.zeros_like(size_parameter_extended),
                where=size_parameter_extended != 0,
            )
        ) * hankel1(p_range + 1 / 2, size_parameter_extended)
        spherical_bessel = spherical_jn(p_range, size_parameter_extended)

        if derivatives:
            spherical_hankel_lower = np.sqrt(
                np.divide(
                    np.pi / 2,
                    size_parameter_extended,
                    out=np.zeros_like(size_parameter_extended),
                    where=size_parameter_extended != 0,
                )
            ) * hankel1(-1 / 2, size_parameter_extended)
            spherical_hankel_lower = np.vstack(
                (spherical_hankel_lower, spherical_hankel[:-1, :, :, :])
            )
            spherical_hankel_derivative = (
                size_parameter_extended * spherical_hankel_lower
                - p_range * spherical_hankel
            )

            # p_range = np.arange(2 * lmax + 2) - 1
            # p_range = p_range[:, np.newaxis, np.newaxis, np.newaxis]
            # spherical_hankel = np.sqrt(np.divide(np.pi / 2, size_parameter_extended, out = np.zeros_like(size_parameter_extended), where = size_parameter_extended != 0)) * hankel1(p_range + 1/2, size_parameter_extended)
            # spherical_hankel_derivative = size_parameter_extended * spherical_hankel[:-1, :, :, :] - p_range[1:, :, :, :] * spherical_hankel[1:, :, :, :]

            p_lm = legendre_normalized_trigon(lmax, cosine_theta, sine_theta)
        else:
            spherical_hankel_derivative = None

            p_lm = np.zeros(
                (lmax + 1) * (2 * lmax + 1) * np.prod(size_parameter.shape[:2])
            ).reshape(((lmax + 1) * (2 * lmax + 1),) + size_parameter.shape[:2])
            for p in range(2 * lmax + 1):
                for absdm in range(p + 1):
                    cml = np.sqrt(
                        (2 * p + 1)
                        / 2
                        * np.prod(1 / np.arange(p - absdm + 1, p + absdm + 1))
                    )
                    # if np.isnan(cml):
                    #     print(p)
                    #     print(absdm)
                    p_lm[p * (p + 1) // 2 + absdm, :, :] = (
                        cml * np.power(-1.0, absdm) * lpmv(absdm, p, cosine_theta)
                    )

        phi = phi[np.newaxis, :, :]
        p_range = np.arange(-2 * lmax, 2 * lmax + 1)
        p_range = p_range[:, np.newaxis, np.newaxis]
        e_j_dm_phi = np.exp(1j * p_range * phi)

    return (
        spherical_bessel,
        spherical_hankel,
        e_j_dm_phi,
        p_lm,
        e_r,
        e_theta,
        e_phi,
        cosine_theta,
        sine_theta,
        size_parameter,
        spherical_hankel_derivative,
    )


import io
import pandas as pd
import yaml
import urllib.request
from urllib.parse import unquote
import requests as req


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
    name = re.split("\._-", path)
    material = unquote(name[0])
    data = pd.read_csv(
        path, delim_whitespace=False, header=0, names=["wavelength", "n", "k"]
    )
    return data, material
