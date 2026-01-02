"""Streamlit data explorer for YASF export files.

This app is intended to be launched via ``yasfpy explore``.
"""

import argparse
import bz2
import importlib
import sys
from pathlib import Path
from typing import Any

import _pickle
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly import colors
from plotly.subplots import make_subplots
from scipy.io import loadmat

try:
    pv: Any = importlib.import_module("pyvista")
except ImportError:  # pragma: no cover
    pv = None

try:
    _stpyvista_mod: Any = importlib.import_module("stpyvista")
    stpyvista = getattr(_stpyvista_mod, "stpyvista")
except ImportError:  # pragma: no cover
    stpyvista = None

if pv is None or stpyvista is None:  # pragma: no cover
    raise ImportError(
        "The Streamlit explorer requires optional dependencies 'pyvista' and 'stpyvista'."
    )

if "IS_XVFB_RUNNING" not in st.session_state:
    pv.start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

CROSS_SECTION_SCALE = 1e6

st.set_page_config(
    page_title="YASF",
    page_icon="\U0001f52c",
    layout="wide",
)
st.title("Yet Another Scattering Framework")


@st.cache_data
def load_data(path: str) -> dict:
    """Load one exported result file.

    Parameters
    ----------
    path:
        Path to an input file. Supported extensions are ``.bz2`` (pickle) and
        ``.mat`` (MATLAB).

    Returns
    -------
    dict
        Parsed export record.
    """

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

with st.sidebar:
    filter_form = st.form("filter-form")
    sources = filter_form.multiselect(
        label="Sources",
        options=list(set([f.stem.split("_")[0] for f in files])),
        default=[],
    )
    configs = filter_form.multiselect(
        label="Config",
        options=list(set([f.stem.split("_")[1] for f in files])),
        default=[],
        format_func=lambda x: " ".join(x.split("-")),
    )
    params = list(
        set(
            [item for f in files for item in "_".join(f.stem.split("_")[2:]).split("-")]
        )
    )
    params.sort()
    params = filter_form.multiselect(
        label="Params",
        options=params,
        default=[],
        format_func=lambda x: ".".join(x.split("_")),
    )
    filter_form.form_submit_button("Filter")

if sources:
    files = [f for f in files if f.stem.split("_")[0] in sources]
if configs:
    files = [f for f in files if f.stem.split("_")[1] in configs]
if params:
    to_remove = []
    for f in files:
        f_params = list(
            set([item for item in "_".join(f.stem.split("_")[2:]).split("-")])
        )
        f_params.sort()
        for p in params:
            if p in f_params:
                break
        else:
            to_remove.append(f)
    for f in to_remove:
        files.remove(f)
files.sort()

with st.sidebar:
    with st.form("file-form"):
        file = st.multiselect(
            label="File",
            options=files,
            format_func=lambda x: x.name,
            help="Resize the sidebar if the paths are cut off",
        )
        submitted = st.form_submit_button("Submit")
if len(files) == 0:
    st.error("No files to be displaye")
if len(file) == 0:
    st.warning("Please select at least one file to be shown")
    st.stop()

dataset = {f.name: load_data(f) for f in file}

first_key = next(iter(dataset))
if "scale" in dataset[first_key]:
    match dataset[first_key]["scale"]:
        case 1:
            length_suffix = "m"
        case 1e-3:
            length_suffix = "mm"
        case 1e-6:
            length_suffix = "\u00b5m"
        case 1e-9:
            length_suffix = "nm"
        case _:
            length_suffix = f"\u2022{dataset[first_key]['scale']}m"
else:
    length_suffix = "\u00b5m"

figs = dict(
    particles={},
    mixing_components=make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
    ),
    angle=go.Figure(),
)

col1, col2, col3 = st.columns(3)
with col1:
    xy_polar = st.toggle(
        label="XY / Polar Plot",
        value=False,
        help="Switch between XY and Polar plot",
    )
with col2:
    log_plot = st.toggle(
        label="Log Plot",
        value=False,
        disabled=xy_polar,
    )
    log_plot &= not xy_polar
with col3:
    scattering_phase_angle = st.toggle(
        label="Scattering Angle / Phase Angle",
        value=True,
    )

plot_type = []
for data in dataset.values():
    current_types = [
        key
        for key in data["angle"].keys()
        if key not in ["theta"]
        and len(data["angle"][key]) + len(data["spatial"][key]) > 0
    ]
    if len(plot_type) == 0:
        plot_type = current_types
    else:
        plot_type = list(set(plot_type) & set(current_types))
plot_type = st.sidebar.selectbox(
    label="Plot Type",
    options=plot_type,
    index=0,
    format_func=lambda x: x.replace("_", " "),
)

cmap_mix = colors.sample_colorscale("Viridis", np.linspace(0, 1, len(dataset)))
with st.sidebar.form("mini-settings"):
    with st.sidebar:
        submitted = st.form_submit_button("Submit")
    for i, (filename, data) in enumerate(dataset.items()):
        if i > 0:
            st.divider()
        st.write(f"**{filename}**")
        st.write(
            f"Radii distribution (common):    {np.mean(data['particles']['radii'])} \u00b1 {np.std(data['particles']['radii'])}"
        )
        st.write(
            f"Radii distribution (geometric): {np.exp(np.mean(np.log(data['particles']['radii'])))} \u00b1 {np.exp(np.std(np.log(data['particles']['radii'])))}"
        )
        for key1 in data.keys():
            if key1 == "source" or key1 == "scale" or key1.startswith("__"):
                continue
            for key2 in data[key1].keys():
                data[key1][key2] = np.array(data[key1][key2])

        if not data["wavelength"]["value"].shape:
            data["wavelength"]["value"] = data["wavelength"]["value"][np.newaxis]
            for key, value in data["angle"].items():
                if key == "theta":
                    continue
                data["angle"][key] = value[:, np.newaxis]
        wavelength_filter = st.multiselect(
            label="Select wavelengths",
            options=np.arange(data["wavelength"]["value"].size),
            default=np.arange(data["wavelength"]["value"].size),
            format_func=lambda x: f"{data['wavelength']['value'][x]}{length_suffix}",
            key=f"multiselect_{filename}",
        )
        wavelength_filter.sort()
        wavelength = np.array(data["wavelength"]["value"])
        if wavelength.size > 1:
            wavelength_slider = st.slider(
                "Wavelength Slider",
                0,
                wavelength.size - 1,
                0,
                1,
                key=f"slider_{filename}",
            )
        else:
            wavelength_slider = 0
        st.write(f"Current wavelength: {wavelength[wavelength_slider]:.2f}&mu;m")

        sampling_points = None
        scattered_field = None
        if "field" in data:
            sampling_points = np.array(data["field"]["sampling_points"]) * 1e-3
            scattered_field = data["field"]["scattered_field"]

        with st.container():
            col1, col2 = (
                st.columns([1, 2])
                if (scattered_field is not None)
                else st.columns([100, 1])
            )
            with col1:
                position = data["particles"]["position"]
                radii = data["particles"]["radii"]
                if "refractive_index" in data["particles"]:
                    unique_indices, refractive_index_label = np.unique(
                        data["particles"]["refractive_index"],
                        return_inverse=True,
                    )
                    n_materials = len(unique_indices)
                else:
                    refractive_index_label = np.zeros(data["particles"]["radii"].size)
                    n_materials = 1
                position -= np.mean(position, axis=0)
                point_cloud = pv.PolyData(position)
                point_cloud["radius"] = [2 * i for i in radii]
                point_cloud["material_index"] = refractive_index_label

                geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
                glyphed = point_cloud.glyph(
                    scale="radius",
                    geom=geom,
                    orient=False,
                )
                pl = pv.Plotter(window_size=[400, 400])
                pl.add_mesh(
                    glyphed,
                    smooth_shading=True,
                    pbr=True,
                    cmap="winter",
                    scalars="material_index",
                    n_colors=n_materials,
                    rng=[0, n_materials - 1],
                )
                pl.view_isometric()
                pl.link_views()
                figs["particles"][filename] = pl
            with col2:
                if (scattered_field is not None) and (sampling_points is not None):
                    eps = np.finfo(float).eps

                    vals = np.linalg.norm(np.abs(scattered_field), axis=2)
                    vals_log = np.log(vals + eps)
                    vals_log_min = np.min(vals_log)
                    vals_log_max = np.max(vals_log)

                    tick_vals_log = np.linspace(vals_log_min, vals_log_max, 15)
                    tick_vals = [f"{x:.2e}" for x in np.exp(tick_vals_log) - eps]

                    fig = go.Figure(
                        data=go.Volume(
                            x=sampling_points[:, 0].flatten(),
                            y=sampling_points[:, 1].flatten(),
                            z=sampling_points[:, 2].flatten(),
                            value=vals_log[wavelength_slider, :],
                            isomin=vals_log_min,
                            isomax=vals_log_max,
                            opacity=0.1,
                            surface_count=15,
                            colorscale="jet",
                            colorbar=dict(
                                tickvals=tick_vals_log,
                                ticktext=tick_vals,
                            ),
                        )
                    )
                    fig.update_layout(
                        height=800,
                        title="Electric Field",
                        scene=dict(
                            xaxis=dict(ticksuffix=length_suffix),
                            yaxis=dict(ticksuffix=length_suffix),
                            zaxis=dict(ticksuffix=length_suffix),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with st.container():
            fig = figs["mixing_components"]
            if isinstance(fig, dict):
                raise Exception(f"Figure {fig} is of type dict...")
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=data["wavelength"]["scattering_efficiency"],
                    name="Q<sub>sca</sub>",
                    line=dict(color=cmap_mix[i]),
                    legendgrouptitle_text=filename,
                    legendgroup=filename,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=data["wavelength"]["extinction_efficiency"],
                    name="Q<sub>ext</sub>",
                    line=dict(color=cmap_mix[i]),
                    legendgrouptitle_text=filename,
                    legendgroup=filename,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=data["wavelength"]["single_scattering_albedo"],
                    name="w",
                    line=dict(color=cmap_mix[i]),
                    legendgrouptitle_text=filename,
                    legendgroup=filename,
                ),
                row=3,
                col=1,
            )

            fig.update_layout(
                title="Mixing components",
                height=900,
                xaxis3=dict(title="Wavelength", ticksuffix="&mu;m"),
                yaxis1=dict(
                    title="Scattering Efficiency",
                    showexponent="all",
                    exponentformat="e",
                ),
                yaxis2=dict(
                    title="Extinction Efficiency",
                    showexponent="all",
                    exponentformat="e",
                ),
                yaxis3=dict(title="Single-Scattering Albedo"),
            )

        with st.container():
            col1, col2 = (
                st.columns(2)
                if data["spatial"]["azimuthal"].size > 0
                else st.columns([100, 1])
            )

            with col1:
                fig = figs["angle"]
                if isinstance(fig, dict):
                    raise Exception(f"Figure {fig} is of type dict...")
                cmap = colors.sample_colorscale(
                    "Jet", np.linspace(0, 1, data["wavelength"]["value"].size)
                )
                theta = data["angle"]["theta"]
                if scattering_phase_angle:
                    theta *= -1
                    theta += 180
                for wavelength_index in wavelength_filter:
                    p = data["angle"][plot_type][:, wavelength_index]
                    if not xy_polar:
                        fig.add_trace(
                            go.Scatter(
                                x=theta,
                                y=p,
                                line=dict(
                                    color=cmap[wavelength_index],
                                    dash=f"{i}px",
                                    width=3,
                                ),
                                name=f"p(\u03b8, {wavelength[wavelength_index]:.2f}{length_suffix})",
                                legendgroup=f"angle_{filename}",
                                legendgrouptitle_text=f"{filename}",
                            )
                        )
                        fig.update_layout(
                            yaxis1=dict(
                                title=plot_type.replace("_", " "),
                                type="log" if log_plot else "-",
                            ),
                        )
                    else:
                        if data["source"] == "yasf":
                            theta = (
                                np.concatenate((theta, 2 * np.pi - theta)) * 180 / np.pi
                            )
                            p = np.concatenate((p, np.flip(p)))
                        fig.add_trace(
                            go.Scatterpolar(
                                theta=theta,
                                r=p,
                                line=dict(color=cmap[wavelength_index]),
                                name="Polar Plot",
                                text=f"\u03bb = {wavelength[wavelength_index]}",
                                legendgrouptitle_text=f"p(\u03b8, {wavelength[wavelength_index]:.2f}nm)",
                                legendgroup=f"group{wavelength_index}",
                            )
                        )
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    dtick=1,
                                    type="log" if log_plot else "-",
                                )
                            ),
                        )
                fig.update_layout(
                    xaxis1=dict(
                        title="Phase Angle"
                        if scattering_phase_angle
                        else "Scattering Angle",
                        ticksuffix="\u00b0",
                        tickmode="linear",
                        tick0=0,
                        dtick=45,
                    ),
                )

            with col2:
                polar_angles = data["spatial"]["polar"]
                azimuthal_angles = data["spatial"]["azimuthal"]
                if (
                    polar_angles.size > 0
                    and azimuthal_angles.size > 0
                    and data["spatial"][plot_type].size > 0
                ):
                    points = np.vstack(
                        [
                            np.sin(polar_angles) * np.cos(azimuthal_angles),
                            np.sin(polar_angles) * np.sin(azimuthal_angles),
                            np.cos(polar_angles),
                        ]
                    ).T

                    p = np.log(data["spatial"][plot_type] + 1)
                    fig = go.Figure(
                        go.Scatter3d(
                            x=points[:, 0] * p[:, wavelength_slider],
                            y=points[:, 1] * p[:, wavelength_slider],
                            z=points[:, 2] * p[:, wavelength_slider],
                            mode="markers",
                            marker=dict(
                                size=1,
                                color=p[:, wavelength_slider],
                                colorscale="Jet",
                                opacity=0.8,
                            ),
                        )
                    )
                    points_extrem = points * np.max(p, axis=1)[:, np.newaxis]
                    fig.update_layout(
                        title="3D representation of the " + plot_type,
                        height=800,
                        scene=dict(
                            xaxis=dict(
                                range=[
                                    np.min(points_extrem[:, 0]),
                                    np.max(points_extrem[:, 0]),
                                ],
                                showticklabels=False,
                            ),
                            yaxis=dict(
                                range=[
                                    np.min(points_extrem[:, 1]),
                                    np.max(points_extrem[:, 1]),
                                ],
                                showticklabels=False,
                            ),
                            zaxis=dict(
                                range=[
                                    np.min(points_extrem[:, 2]),
                                    np.max(points_extrem[:, 2]),
                                ],
                                showticklabels=False,
                            ),
                            aspectratio=dict(x=1, y=1, z=1),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)


for fig in figs.values():
    if isinstance(fig, dict):
        for key, val in fig.items():
            st.write(key)
            stpyvista(val)
    else:
        st.plotly_chart(fig, use_container_width=True)

st.write("**Raw Data**")
for filename, data in dataset.items():
    with st.expander(filename):
        st.write(data)
