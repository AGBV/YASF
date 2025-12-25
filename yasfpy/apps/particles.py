import argparse
import bz2
import re
import sys
from io import BytesIO
from pathlib import Path

import _pickle
import numpy as np
import pyvista as pv
import streamlit as st
from scipy.io import loadmat
from stpyvista import stpyvista

if "IS_XVFB_RUNNING" not in st.session_state:
    pv.start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

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


parser = argparse.ArgumentParser(description="Particle Display")
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
    all_files = st.checkbox("Select all files", value=True)

files.sort()
with st.sidebar:
    with st.form("file-form"):
        file = st.multiselect(
            label="File(s)",
            options=files,
            # default=files[0],
            format_func=lambda x: x.name,
            help="Resize the sidebar if the paths are cut off",
            disabled=all_files,
        )
        submitted = st.form_submit_button("Submit", disabled=all_files)
    if all_files:
        file = files
    link_views = st.checkbox(
        label="Link views",
        value=True,
        help="Link the views of all plots, i.e., pan and zoom is synced",
    )
    font_size = st.number_input(
        label="Font size",
        min_value=6,
        # max_value=24,
        value=12,
        step=1,
        help="Font size for titles and labels",
    )
    scale = st.number_input(
        label="Scale",
        min_value=1,
        # max_value=10,
        value=4,
        step=1,
        help="Scale of the plot screenshot",
    )
    columns = st.number_input(
        label="Columns",
        min_value=1,
        # max_value=10,
        value=int(np.ceil(np.sqrt(len(file)))),
        step=1,
        help="Number of columns in the plot grid",
    )

sort_keys = {}
for f in file:
    m = re.findall(r"N(\d+)|Df([p\d]+)", f.name)
    if not m:
        st.warning(f"File {f.name} does not match expected pattern")
        continue
    sort_keys[f.name] = dict(
        N=int(m[0][0]),
        Df=float(m[1][1].replace("p", ".")),
    )
file.sort(key=lambda x: (sort_keys[x.name]["N"], sort_keys[x.name]["Df"]))

# n = int(np.ceil(np.sqrt(len(file))))
rows = int(len(file) // columns)
st.write(rows, columns)
pl = pv.Plotter(
    window_size=[300 * columns, 300 * rows],
    shape=(int(len(file) // columns), columns),
    off_screen=True,
)
geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
for k, f in enumerate(file):
    i = k // columns
    j = k % columns
    match f.suffix:
        case ".dat":
            data = np.loadtxt(f)
            # m = re.findall(
            #     r"N(\d+)|Df([p\d]+)",
            #     f.name,
            # )
            # df = float(m[1][1].replace("p", "."))
            # N = int(m[0][0])
        case _:
            raise Exception("Unsupported file type for particles display")
    radii = data[:, 3]
    position = data[:, :3]
    position -= np.mean(position, axis=0)
    point_cloud = pv.PolyData(position)
    point_cloud["radius"] = [2 * i for i in radii]
    glyphed = point_cloud.glyph(
        scale="radius",  # type: ignore
        geom=geom,
        orient=False,
    )
    pl.subplot(i, j)
    pl.add_title(
        f"N={sort_keys[f.name]['N']} | Df={sort_keys[f.name]['Df']:.2f}",
        font_size=font_size,
    )
    pl.add_mesh(
        glyphed,
        color="white",
        smooth_shading=True,
        pbr=True,
        cmap="winter",
        # scalars="material_index",
        # n_colors=n_materials,
        # rng=[0, n_materials - 1],
    )
    pl.view_isometric()  # type: ignore
    if link_views:
        pl.link_views()
st.write(stpyvista(pl))

plot_data = BytesIO()
# pl.show(screenshot=plot_data)
pl.screenshot(plot_data, transparent_background=True, scale=scale)

with st.sidebar:
    st.download_button(
        label="Download Plot",
        data=plot_data,
        file_name="plot.png",
        mime="image/png",
        icon=":material/download:",
        on_click="ignore",
    )
