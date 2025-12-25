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
def load_data(path: str) -> dict:
    p = Path(path)
    match p.suffix:
        case ".bz2":
            with bz2.BZ2File(path, "rb") as file:
                data = _pickle.load(file)
        case ".mat":
            data = loadmat(path, simplify_cells=True)
        case _:
            raise Exception(f"Unknown file type: {p.suffix}")
    return data


# @jit
def minimal_touching_circles_quadrature(x_centers: npt.NDArray, realistic: bool = True):
    """
    Calculates the radii of circles centered at x_centers such that adjacent
    circles touch at one point (no overlap or gap) and the total area of
    the circles is minimized.

    The circles are defined by centers x_1, x_2, ..., x_n and radii
    r_1, r_2, ..., r_n. The tangency constraint is x_{i+1} - x_i = r_i + r_{i+1}.
    The minimum area constraint leads to sum_{i=1}^n (-1)^(i-1) * r_i = 0.

    Args:
        x_centers (array-like): A 1D array or list of the x-coordinates
                                of the circle centers. Must contain at least
                                two points. Input points need not be sorted.
        realistic (bool): If False, negative radii can occure in the result.
                          If True, a heuristic will be used for positive radii.

    Returns:
        numpy.ndarray: A 1D array containing the calculated radius for each
                       circle (r_1, r_2, ..., r_n), corresponding to the
                       original order of x_centers. Returns None if calculation
                       results in non-physical (negative) radii.

    Raises:
        ValueError: If x_centers contains fewer than two points, is not 1D,
                    or contains duplicate values.
    """
    # --- Input Validation and Preparation ---
    x = np.asarray(x_centers, dtype=float)  # Ensure float array for calculations

    if x.ndim != 1:
        raise ValueError("Input x_centers must be a 1D array.")
    n = len(x)
    if n < 2:
        raise ValueError("Input x_centers must contain at least two points.")

    # Sort points and keep track of the sorting order
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]

    # --- Calculate Distances Between Sorted Points ---
    # d[k] corresponds to d_{k+1} = x_sorted[k+1] - x_sorted[k]
    d = np.diff(x_sorted)

    # Check for duplicate points after sorting
    if np.any(d <= 0):
        raise ValueError("Input x_centers must have unique, distinct values.")

    # --- Calculate r_1 (for the sorted sequence) using the minimum area formula ---
    # Formula: r_1_sorted = (1/n) * sum_{k=1}^{n-1} (n - k) * (-1)**(k+1) * d_k
    # In NumPy terms (d array is 0-indexed, so d[k-1] is d_k):
    k = np.arange(1, n)  # k = 1, 2, ..., n-1
    weights = n - k  # (n-1), (n-2), ..., 1
    signs = (-1) ** (k + 1)  # +1, -1, +1, ... corresponding to k=1, 2, 3...
    s = d * signs

    # The k-th term in the sum uses d_k, which is d[k-1] in the NumPy array d
    if realistic:
        s_cumsum = np.cumsum(s)
        lower = np.max(s_cumsum[1::2])
        upper = np.min(s_cumsum[::2])
        if lower <= upper:
            r1_sorted = (max(lower, 0) + max(upper, 0)) / n  # Avoid negative radius
        else:
            st.warning("No sphere insertion possible. Using a simple heuristic.")
            r_sorted = (
                np.hstack(
                    (
                        x_sorted[1] - x_sorted[0],
                        x_sorted[2:] - x_sorted[:-2],
                        x_sorted[-1] - x_sorted[-2],
                    )
                )
                / 2
            )
            r_final = np.zeros(n)
            r_final[sort_indices] = r_sorted
            return r_final
    else:
        r1_sorted = np.sum(weights * s) / n  # Float division

    # --- Calculate Remaining Radii Iteratively (for the sorted sequence) ---
    r_sorted = np.zeros(n)
    r_sorted[0] = r1_sorted

    # r_sorted[i] = d[i-1] - r_sorted[i-1]  (corresponds to r_i = d_{i-1} - r_{i-1})
    for i in k:
        r_sorted[i] = d[i - 1] - r_sorted[i - 1]
    # --- Final Checks ---
    # Check for negative radii (within a small tolerance for floating point errors)
    # A negative radius indicates the minimum area solution is not physically possible
    # with all circles having positive radius under the tangency constraints.
    # tolerance = -1e-12
    # if np.any(r_sorted < tolerance):
    #     neg_indices_sorted = np.where(r_sorted < tolerance)[0]
    #     # Map back to original indices for warning message
    #     neg_indices_original = sort_indices[neg_indices_sorted]
    #     print(
    #         f"Warning: Negative radius calculated for input point index(es) "
    #         f"{neg_indices_original} (value(s): {r_sorted[neg_indices_sorted]}). "
    #         f"This setup does not allow for a physical solution with all positive radii "
    #         f"satisfying tangency and minimum area. Returning None."
    #     )
    #     return None  # Indicate failure

    # --- Reorder Radii to Match Original Input Order ---
    # Create an array to store the final radii in the original order
    r_final = np.zeros(n)
    # Place the calculated sorted radii back into their original positions
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
except SystemExit as e:
    sys.exit(e.code)

files = []
for path in args.path:
    p = Path(path)
    if not p.exists():
        st.warning(f"Path {p.resolve()} does not exist")
        continue
    files.extend([item for item in p.rglob("*") if item.is_file()])

sources: list[Any] = [None] * NUMBER_OF_SOURCES
configs: list[Any] = [None] * NUMBER_OF_SOURCES
params: list[Any] = [None] * NUMBER_OF_SOURCES
all_files: list[bool] = [True] * NUMBER_OF_SOURCES
file: list[list[Path]] = [[Path()]] * NUMBER_OF_SOURCES
for i in range(NUMBER_OF_SOURCES):
    with st.sidebar:
        if i > 0:
            st.divider()
        st.write(f"Dataset {i + 1}")
        filter_form = st.form(f"filter-form-{i}")
        sources[i] = filter_form.multiselect(
            label="Sources",
            options=list(set([f.stem.split("_")[0] for f in files])),
            default=[],
        )
        configs[i] = filter_form.multiselect(
            label="Config",
            options=list(set([f.stem.split("_")[1] for f in files])),
            default=[],
            format_func=lambda x: " ".join(x.split("-")),
        )
        params_temp = []
        for p in ["_".join(f.stem.split("_")[2:]) for f in files]:
            if p.startswith("fracval"):
                elements = p.split("_")
                elements = [e for e in elements if not e.startswith("agg")]
                elements = [e for e in elements if not e[0].isdigit()]
                params_temp.extend(elements)
            else:
                params_temp.extend(p.split("-"))
        params_temp = list(set([item for item in params_temp]))
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
        to_remove = []
        for f in files_copy:
            f_params = "_".join(f.stem.split("_")[2:])
            if f_params.startswith("fracval"):
                f_params = list(set([item for item in f_params.split("_")]))
            else:
                f_params = list(set([item for item in f_params.split("-")]))
            f_params.sort()
            for p in params[i]:
                if p in f_params:
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
                # default=files[0],
                format_func=lambda x: x.name,
                help="Resize the sidebar if the paths are cut off",
                disabled=all_files[i],
            )
            submitted = st.form_submit_button("Submit", disabled=all_files[i])
        if all_files[i]:
            file[i] = files_copy

    # if len(files_copy) == 0:
    #     st.error("No files to be displayed")
    #     st.stop()
datasets = [
    {f.name: load_data(f.resolve().as_posix()) for f in file[i]}
    for i in range(NUMBER_OF_SOURCES)
]

for i in range(NUMBER_OF_SOURCES):
    # print(file[i])
    if len(file[i]) == 0:
        st.warning("Please select at least one file to be shown")
        # st.stop()

# dataset = datasets[0]
# print(f"{dataset.keys()=}")

theta = np.linspace(0, 180, 181)
dolp_interpolated: list[dict[str, Any]] = [{}] * NUMBER_OF_SOURCES
for dataset_index, dataset in enumerate(datasets):
    if not dataset:
        continue
    for key, value in dataset.items():
        # if "metadata" in value["particles"]:
        #     print(
        #         value["particles"]["metadata"]["aggregate_properties"]["radius_of_gyration"]
        #         / 1000
        #     )
        if "radius_of_gyration" in value["particles"]:
            # print(value["particles"]["radius_of_gyration"])
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
            # marker=dict(size=1),
            # name=file[0],
        )
    )
    for r, q in zip(rog, quadrature, strict=True):
        fig.add_shape(
            type="rect",
            # xref="x",
            # yref="y",
            x0=r - q,
            y0=0,
            x1=r + q,
            y1=r**n,
            fillcolor="LightSkyBlue",
            opacity=0.2,
        )
    # fig.update_layout(yaxis=dict(type="log"))
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
    wavelengths_to_pick = set()
    for key, value in dataset.items():
        if isinstance(value["wavelength"]["value"], float):
            dataset[key]["wavelength"]["value"] = np.array(
                [value["wavelength"]["value"]]
            )
            dataset[key]["angle"]["degree_of_linear_polarization"] = value["angle"][
                "degree_of_linear_polarization"
            ][:, np.newaxis]
        wavelengths_to_pick.update(value["wavelength"]["value"].tolist())
    # TODO: move this up somehow...
    wavelength_pick = st.sidebar.selectbox(
        "Pick a wavelength",
        wavelengths_to_pick,
        key=f"wavelength-{dataset_index}",
    )

    dolp = np.empty(shape=(theta.size, 0))
    dfs = np.empty(shape=0)
    matching_files = []
    for i, (key, value) in enumerate(dataset.items()):
        idx = np.where(value["wavelength"]["value"] == wavelength_pick)[0]
        if len(idx) == 0:
            weights[i] = np.nan
            quadrature[i] = np.nan
            continue
        matching_files.append(key)
        idx = idx[0]
        dolp = np.hstack(
            (
                dolp,
                np.flip(
                    np.interp(
                        theta,
                        value["angle"]["theta"],
                        value["angle"]["degree_of_linear_polarization"][:, idx],
                    )
                )[:, np.newaxis],
            ),
        )
        if "metadata" in value["particles"]:
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
        local_data = {}
        for df in unique_df:
            data_average = (
                np.average(dolp, axis=1, weights=weights * quadrature * (df == dfs)),
            )
            local_data[df] = data_average[0]
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

result = {}
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
