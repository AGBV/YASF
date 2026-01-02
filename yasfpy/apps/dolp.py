"""Streamlit visualization for polarization diagnostics.

This app provides an interactive explorer for previously computed YASF datasets.
It focuses on the degree of linear polarization (DoLP) as a function of scattering
angle and allows comparing two datasets.

Notes
-----
This module is executed by Streamlit. Most code runs at import time.
"""

import argparse
import bz2
import sys
from pathlib import Path
from typing import Any

import _pickle
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import streamlit as st
from scipy.io import loadmat

from yasfpy.particles import radius_of_gyration

NUMBER_OF_SOURCES = 2

st.set_page_config(layout="wide")


@st.cache_data
def load_data(path: str) -> dict[str, Any]:
    """Load a YASF result file.

    Parameters
    ----------
    path:
        Path to a data file. Supported extensions are ``.bz2`` (pickled) and
        ``.mat`` (MATLAB).

    Returns
    -------
    dict[str, Any]
        Decoded data dictionary.
    """

    file_path = Path(path)
    match file_path.suffix:
        case ".bz2":
            with bz2.BZ2File(path, "rb") as file:
                data = _pickle.load(file)
        case ".mat":
            data = loadmat(path, simplify_cells=True)
        case _:
            raise ValueError(f"Unknown file type: {file_path.suffix}")
    return data


def minimal_touching_circles_quadrature(
    x_centers: npt.ArrayLike,
    realistic: bool = True,
) -> npt.NDArray[np.floating[Any]]:
    """Compute a 1D "touching circles" quadrature rule.

    Given circle centers :math:`x_1 < x_2 < \dots < x_n`, this computes radii
    :math:`r_1, \dots, r_n` such that adjacent circles are tangent
    (:math:`x_{i+1}-x_i = r_i + r_{i+1}`) and the total area is minimized.

    Parameters
    ----------
    x_centers:
        1D coordinates of the circle centers.
    realistic:
        If ``True``, apply a heuristic to avoid negative radii.

    Returns
    -------
    numpy.ndarray
        Radii in the original input order.

    Raises
    ------
    ValueError
        If the input array is not 1D, contains fewer than two points, or has
        duplicates.
    """

    x = np.asarray(x_centers, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input x_centers must be a 1D array.")

    n = len(x)
    if n < 2:
        raise ValueError("Input x_centers must contain at least two points.")

    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]

    d = np.diff(x_sorted)
    if np.any(d <= 0):
        raise ValueError("Input x_centers must have unique, distinct values.")

    k = np.arange(1, n)
    weights = n - k
    signs = (-1) ** (k + 1)
    s = d * signs

    if realistic:
        s_cumsum = np.cumsum(s)
        lower = np.max(s_cumsum[1::2])
        upper = np.min(s_cumsum[::2])

        if lower <= upper:
            r1_sorted = (max(lower, 0) + max(upper, 0)) / n
        else:
            st.warning("No sphere insertion possible. Using a simple heuristic.")
            r_sorted_simple = (
                np.hstack(
                    (
                        x_sorted[1] - x_sorted[0],
                        x_sorted[2:] - x_sorted[:-2],
                        x_sorted[-1] - x_sorted[-2],
                    )
                )
                / 2
            )
            r_final_simple = np.zeros(n)
            r_final_simple[sort_indices] = r_sorted_simple
            return r_final_simple
    else:
        r1_sorted = np.sum(weights * s) / n

    r_sorted = np.zeros(n)
    r_sorted[0] = r1_sorted

    for i in k:
        r_sorted[i] = d[i - 1] - r_sorted[i - 1]

    r_final = np.zeros(n)
    r_final[sort_indices] = r_sorted
    return r_final


parser = argparse.ArgumentParser(description="Data explorer for YASF")
parser.add_argument(
    "--path",
    action="append",
    default=[],
    help="Set path to look for data",
)

try:
    args = parser.parse_args()
except SystemExit as exc:
    sys.exit(exc.code)

files: list[Path] = []
for path in args.path:
    p = Path(path)
    if not p.exists():
        st.warning(f"Path {p.resolve()} does not exist")
        continue
    files.extend([item for item in p.rglob("*") if item.is_file()])

sources: list[list[str]] = [[] for _ in range(NUMBER_OF_SOURCES)]
configs: list[list[str]] = [[] for _ in range(NUMBER_OF_SOURCES)]
params: list[list[str]] = [[] for _ in range(NUMBER_OF_SOURCES)]
all_files: list[bool] = [True] * NUMBER_OF_SOURCES
file: list[list[Path]] = [[] for _ in range(NUMBER_OF_SOURCES)]

for i in range(NUMBER_OF_SOURCES):
    with st.sidebar:
        if i > 0:
            st.divider()
        st.write(f"Dataset {i + 1}")

        filter_form = st.form(f"filter-form-{i}")
        sources[i] = filter_form.multiselect(
            label="Sources",
            options=list({f.stem.split("_")[0] for f in files}),
            default=[],
        )
        configs[i] = filter_form.multiselect(
            label="Config",
            options=list({f.stem.split("_")[1] for f in files}),
            default=[],
            format_func=lambda x: " ".join(x.split("-")),
        )
        params_temp: list[str] = []
        for p in ["_".join(f.stem.split("_")[2:]) for f in files]:
            if p.startswith("fracval"):
                elements = p.split("_")
                elements = [e for e in elements if not e.startswith("agg")]
                elements = [e for e in elements if not e[0].isdigit()]
                params_temp.extend(elements)
            else:
                params_temp.extend(p.split("-"))
        params_temp = list({item for item in params_temp})
        params_temp.sort()
        params[i] = filter_form.multiselect(
            label="Params",
            options=params_temp,
            default=[],
            format_func=lambda x: ".".join(x.split("_")),
        )
        all_files[i] = filter_form.checkbox("Select all files", value=False)
        filter_form.form_submit_button("Filter")

    files_copy = files.copy()
    if sources[i]:
        files_copy = [f for f in files_copy if f.stem.split("_")[0] in sources[i]]
    if configs[i]:
        files_copy = [f for f in files_copy if f.stem.split("_")[1] in configs[i]]
    if params[i]:
        to_remove: list[Path] = []
        for f in files_copy:
            f_params = "_".join(f.stem.split("_")[2:])
            if f_params.startswith("fracval"):
                f_params_list = list({item for item in f_params.split("_")})
            else:
                f_params_list = list({item for item in f_params.split("-")})
            f_params_list.sort()

            for p in params[i]:
                if p in f_params_list:
                    break
            else:
                to_remove.append(f)

        for f in to_remove:
            files_copy.remove(f)

    files_copy.sort()

    with st.sidebar:
        with st.form(f"file-form-{i}"):
            file[i] = st.multiselect(
                label="File(s)",
                options=files_copy,
                format_func=lambda x: x.name,
                help="Resize the sidebar if the paths are cut off",
                disabled=all_files[i],
            )
            st.form_submit_button("Submit", disabled=all_files[i])
        if all_files[i]:
            file[i] = files_copy


datasets: list[dict[str, dict[str, Any]]] = [
    {f.name: load_data(f.resolve().as_posix()) for f in file[i]}
    for i in range(NUMBER_OF_SOURCES)
]

for i in range(NUMBER_OF_SOURCES):
    if len(file[i]) == 0:
        st.warning("Please select at least one file to be shown")


theta = np.linspace(0, 180, 181)

dolp_interpolated: list[dict[float, npt.NDArray[np.floating[Any]]]] = [
    {} for _ in range(NUMBER_OF_SOURCES)
]

for dataset_index, dataset in enumerate(datasets):
    if not dataset:
        continue

    for key, value in dataset.items():
        if "radius_of_gyration" in value.get("particles", {}):
            continue

        dataset[key]["particles"]["radius_of_gyration"] = radius_of_gyration(
            value["particles"]["position"],
            value["particles"]["radii"],
        )

    st.write(f"**Dataset {dataset_index + 1}**")
    n = st.number_input(
        "Select power law exponent",
        min_value=-10,
        max_value=-1,
        value=-2,
        step=1,
        format="%d",
        key=f"power-law-exponent-{dataset_index}",
    )

    rog = np.array([v["particles"]["radius_of_gyration"] for v in dataset.values()])
    weights = np.power(rog, n)
    quadrature = minimal_touching_circles_quadrature(rog, realistic=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rog,
            y=weights,
            hovertemplate="""
            <b>x</b>: %{x} <br>
            <b>y</b>: %{y} <br>
            <b>%{text}</b>""",
            text=[
                f"N {len(dataset[k]['particles']['radii'])}, Df {dataset[k]['particles']['metadata']['simulation_parameters']['Df']}"
                for k in dataset.keys()
            ],
            mode="markers",
        )
    )

    for r, q in zip(rog, quadrature, strict=True):
        fig.add_shape(
            type="rect",
            x0=r - q,
            y0=0,
            x1=r + q,
            y1=r**n,
            fillcolor="LightSkyBlue",
            opacity=0.2,
        )

    with st.expander("Radius of Gyration Distribution"):
        st.plotly_chart(fig, key=f"rog-distribution-plot-{dataset_index}")

    unique_df = np.array(
        list(
            {
                value["particles"]["metadata"]["simulation_parameters"]["Df"]
                for value in dataset.values()
            }
        )
    )
    unique_df.sort()

    wavelengths_to_pick: set[float] = set()
    for key, value in dataset.items():
        if isinstance(value["wavelength"]["value"], float):
            dataset[key]["wavelength"]["value"] = np.array(
                [value["wavelength"]["value"]]
            )
            dataset[key]["angle"]["degree_of_linear_polarization"] = value["angle"][
                "degree_of_linear_polarization"
            ][:, np.newaxis]
        wavelengths_to_pick.update(value["wavelength"]["value"].tolist())

    wavelength_options = sorted(wavelengths_to_pick)
    if not wavelength_options:
        st.warning("No wavelengths available for the selected dataset")
        continue

    wavelength_pick = st.sidebar.selectbox(
        "Pick a wavelength",
        wavelength_options,
        key=f"wavelength-{dataset_index}",
    )

    dolp = np.empty(shape=(theta.size, 0))
    dfs = np.empty(shape=0)
    matching_files: list[str] = []

    for i, (key, value) in enumerate(dataset.items()):
        idx = np.where(value["wavelength"]["value"] == wavelength_pick)[0]
        if len(idx) == 0:
            weights[i] = np.nan
            quadrature[i] = np.nan
            continue

        matching_files.append(key)
        idx0 = int(idx[0])

        dolp = np.hstack(
            (
                dolp,
                np.flip(
                    np.interp(
                        theta,
                        value["angle"]["theta"],
                        value["angle"]["degree_of_linear_polarization"][:, idx0],
                    )
                )[:, np.newaxis],
            )
        )

        if "metadata" in value.get("particles", {}):
            dfs = np.append(
                dfs, value["particles"]["metadata"]["simulation_parameters"]["Df"]
            )

    weights = weights[~np.isnan(weights)]
    quadrature = quadrature[~np.isnan(quadrature)]

    fig = go.Figure()
    if dfs.size == 0:
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=np.average(dolp, axis=1, weights=weights * quadrature),
            )
        )
    else:
        local_data: dict[float, npt.NDArray[np.floating[Any]]] = {}
        for df in unique_df:
            data_average = np.average(
                dolp,
                axis=1,
                weights=weights * quadrature * (df == dfs),
            )
            local_data[float(df)] = data_average
            fig.add_trace(
                go.Scatter(
                    x=theta,
                    y=data_average,
                    name=f"Df: {df}",
                )
            )
        dolp_interpolated[dataset_index] = local_data

    with st.expander("DoLP Average(s)"):
        st.plotly_chart(fig)

    with st.expander("List of files with matching wavelength"):
        matching_files.sort()
        st.table(matching_files)


wight = st.slider("Weight Dataset 1", 0.0, 1.0, 0.5)
weights = np.array([wight, 1 - wight])

result: dict[float, npt.NDArray[np.floating[Any]]] = {}
for i, data in enumerate(dolp_interpolated):
    for key, value in data.items():
        if key not in result:
            result[key] = value * weights[i]
        else:
            result[key] += value * weights[i]

fig = go.Figure()
for key, value in result.items():
    fig.add_trace(
        go.Scatter(
            x=theta,
            y=value,
            name=f"Df: {key}",
        )
    )
st.plotly_chart(fig, use_container_width=True)
