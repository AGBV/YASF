import copy
import logging
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyperf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

project_root = Path(__file__).parent.parent.parent
if __name__ == "__main__":
    sys.path.append(str(project_root.absolute()))

from yasfpy.config import Config
from yasfpy.export import Export
from yasfpy.particles import radius_of_gyration


class MSTM4Manager(BaseModel):
    path_config: str = Field(default="")
    path_cluster: str = Field(default="")
    cluster_scale: float = Field(default=1.0)
    cluster_dimensional_scale: float = Field(default=1.0)
    binary: str = Field(default="mstm")
    input_file: str = Field(default="mstm4.inp")
    output_file: str = Field(default="mstm4.dat")
    parallel: int = Field(default=4, multiple_of=4)
    nix: bool = Field(default=True)

    print_sphere_data: bool = Field(default=False)
    random_orientation: bool = Field(default=False)
    incidence_average: bool = Field(default=False)
    number_incident_directions: int = Field(default=100, gt=0)
    scattering_map_model: bool = Field(default=False)
    azimuthal_average: bool = Field(default=False)

    _ref_idx: npt.NDArray = PrivateAttr()
    log: logging.Logger | None = None
    config: Config | None = None
    output: dict = {}
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.log is None:
            self.log = logging.getLogger(self.__class__.__module__)
        self.config = Config(
            path_config=self.path_config,
            path_cluster=self.path_cluster,
            cluster_scale=self.cluster_scale,
            cluster_dimensional_scale=self.cluster_dimensional_scale,
        )
        if self.random_orientation and self.incidence_average:
            self.log.warning("Random orientation and incidence average are both set!")
            self.log.warning("Random orientation will be turned off")
            self.random_orientation = False

    def __write(
        self,
        fixed_lmax: bool = True,
    ) -> str:
        if self.config is None:
            raise Exception("Config is None")

        mstm_config = f"""
        output_file
        {self.output_file}
        print_sphere_data
        {"t" if self.print_sphere_data else "f"}
        calculate_scattering_matrix
        t
        single_origin_expansion
        t
        normalize_s11
        t
        random_orientation
        {"t" if self.random_orientation else "f"}
        incidence_average
        {"t" if self.incidence_average else "f"}
        number_incident_directions
        {self.number_incident_directions:d}
        azimuthal_average
        {"t" if self.azimuthal_average else "f"}
        scattering_map_model
        {"1" if self.scattering_map_model else "0"}
        scattering_map_increment
        1.d0
        incident_beta_deg
        {self.config.config["initial_field"]["polar_angle"]:e}
        incident_alpha_deg
        {self.config.config["initial_field"]["azimuthal_angle"]:e}
        """
        mstm_config = "\n".join(
            [line.strip() for line in mstm_config.split("\n") if line]
        )

        # if fixed_lmax and "lmax" in self.config.config["numerics"]:
        #     mstm_config += "mie_epsilon\n"
        #     mstm_config += f"-{int(self.config.config['numerics']['lmax'])}\n"

        self._ref_idx = np.take(
            self.config.refractive_index_interpolated,
            self.config.spheres[:, 4].astype(int),
            axis=0,
        )
        for wl_idx, wl in enumerate(self.config.wavelength):
            mstm_config += "length_scale_factor\n"
            mstm_config += f"{2 * np.pi / wl}\n"
            mstm_config += "layer_ref_index\n"
            mstm_config += f"({self.config.medium_refractive_index[wl_idx].real}, {self.config.medium_refractive_index[wl_idx].imag})\n"
            mstm_config += "number_spheres\n"
            mstm_config += f"{self.config.spheres.shape[0]}\n"

            mstm_config += "sphere_data\n"
            for particle_idx in range(self.config.spheres.shape[0]):
                position_str = ",".join(
                    [f"{i:e}" for i in self.config.spheres[particle_idx, :-1]]
                )
                ref_idx_str = f"({self._ref_idx[particle_idx, wl_idx].real}, {self._ref_idx[particle_idx, wl_idx].imag})"
                mstm_config += f"{position_str},{ref_idx_str}\n"

            mstm_config += "end_of_sphere_data\n"

            if wl_idx + 1 == self.config.wavelength.size:
                mstm_config += "end_of_options"
            else:
                mstm_config += "new_run\n"

        with open(self.input_file, "w") as fh:
            fh.write(mstm_config)

        return mstm_config

    def __exec(self, runner: pyperf.Runner | None = None, silent: bool = False):
        if self.nix:
            command = [
                self.binary,
                str(self.parallel),
                self.input_file,
            ]
        else:
            command = [
                "mpiexec",
                "-n",
                str(self.parallel),
                self.binary,
                self.input_file,
            ]

        if runner is None:
            if silent:
                subprocess.check_output(" ".join(command), shell=True)
            else:
                os.system(" ".join(command))
        else:
            runner.bench_command(f"mstm4_exec_{self.parallel}", command)

    @staticmethod
    def __parse_input(input):
        # Get id of run
        p = re.compile(r"\s+input variables for run\s+(\d+)")
        r = p.search(input)
        if r is None:
            raise Exception("No run id found in the input file!")
        id = r.group(1)
        id = int(id)

        # lengths scale factor
        p = re.compile(r"\s+length, ref index scale factors\s+(.+?)\s+(.+?)\s+(.+?)")
        r = p.search(input)
        if r is None:
            raise Exception("No length scale factor found in the input file!")
        length_scale_factor = r.group(1)
        length_scale_factor = float(length_scale_factor)

        return dict(
            id=id,
            length_scale_factor=length_scale_factor,
        )

    # @staticmethod
    def __parse_output(self, results):
        # Get id of run
        p = re.compile(r"\s+calculation results for run\s+(\d+)")
        r = p.search(results)
        if r is None:
            raise Exception("No run id found in the output file!")
        id = r.group(1)
        id = int(id)

        sphere_data = None
        if self.print_sphere_data:
            p = re.compile(r"sphere\s+Qext\s+Qabs\s+Qvabs\s+(.+?)total", re.S)
            r = p.search(results)
            if r is None:
                raise Exception("No sphere data found in the output file!")
            sphere_data = r.group(1)
            sphere_data = pd.read_csv(
                StringIO(sphere_data),
                names=["sphere", "Qext", "Qabs", "Qvabs"],
                header=None,
                sep=r"\s+",
            ).drop(columns=["sphere"])

        if self.random_orientation or self.incidence_average:
            p = re.compile(
                r"total extinction, absorption, scattering efficiencies \(unpolarized incidence\)\s+(.+)"
            )
            r = p.search(results)
            if r is None:
                raise Exception("No efficiency data found in the output file!")
            efficiencies = r.group(1)
            efficiencies = [float(e) for e in efficiencies.split()]
            efficiencies = dict(
                q_ext_unp=efficiencies[0],
                q_abs_unp=efficiencies[1],
                q_sca_unp=efficiencies[2],
                q_ext_par=None,
                q_abs_par=None,
                q_sca_par=None,
                q_ext_per=None,
                q_abs_per=None,
                q_sca_per=None,
            )
        else:
            p = re.compile(
                r"total extinction, absorption, scattering efficiencies \(unpol, par, perp incidence\)\s+(.+)"
            )
            r = p.search(results)
            if r is None:
                raise Exception("No efficiency data found in the output file!")
            efficiencies = r.group(1)
            efficiencies = [float(e) for e in efficiencies.split()]
            efficiencies = dict(
                q_ext_unp=efficiencies[0],
                q_abs_unp=efficiencies[1],
                q_sca_unp=efficiencies[2],
                q_ext_par=efficiencies[3],
                q_abs_par=efficiencies[4],
                q_sca_par=efficiencies[5],
                q_ext_per=efficiencies[6],
                q_abs_per=efficiencies[7],
                q_sca_per=efficiencies[8],
            )

        if not self.random_orientation:
            p = re.compile(
                r"down and up hemispherical scattering efficiencies \(unpol, par, perp\)\s+(.+)"
            )
            r = p.search(results)
            if r is None:
                raise Exception(
                    "No hemispherical efficiency data found in the output file!"
                )
            efficiencies_hem = r.group(1)
            efficiencies_hem = [float(e) for e in efficiencies_hem.split()]
        else:
            efficiencies_hem = [None] * 6
        efficiencies_hem = dict(
            q_hem_u_unp=efficiencies_hem[0],
            q_hem_d_unp=efficiencies_hem[1],
            q_hem_u_par=efficiencies_hem[2],
            q_hem_d_par=efficiencies_hem[3],
            q_hem_u_per=efficiencies_hem[4],
            q_hem_d_per=efficiencies_hem[5],
        )

        # p = (
        #     re.compile(
        #         r"theta\s+11\s+12\s+22\s+33\s+34\s+44\s+(.+?)\s+(azimuthal|diffuse|orientation)",
        #         re.S,
        #     )
        #     if self.random_orientation
        #     or self.azimuthal_average
        #     or self.incidence_average
        #     else re.compile(
        #         r"theta\s+11\s+12\s+13\s+14\s+21\s+22\s+23\s+24\s+31\s+32\s+33\s+34\s+41\s+42\s+43\s+44\s+(.+)",
        #         re.S,
        #     )
        # )
        p = re.compile(
            r"((?: +theta| +kx| +ky)+(?:\s+[1-4]{2}){6,16}[\s\d\.\+-E]+)",
            re.M,
        )
        r = p.findall(results)
        if r is None or len(r) == 0:
            raise Exception("No scattering matrix found in the output file!")

        scattering_matrix = pd.read_csv(
            StringIO(r[0]),
            sep=r"\s+",
        )
        if self.scattering_map_model:
            if len(r) < 2:
                raise Exception(
                    "For scattering map model =1, two scattering matrices are required"
                )
            scattering_matrix_lower = pd.read_csv(
                StringIO(r[1]),
                sep=r"\s+",
            )
            for i, sm in enumerate([scattering_matrix, scattering_matrix_lower]):
                kx = sm["kx"].to_numpy()
                ky = sm["ky"].to_numpy()
                cos_theta = np.sqrt(1 - kx**2 - ky**2) * np.power(-1, i)
                theta = np.rad2deg(np.arccos(cos_theta))
                phi = np.rad2deg(np.arctan2(ky, kx))
                sm.rename(columns=dict(kx="theta", ky="phi"), inplace=True)
                sm["theta"] = theta
                sm["phi"] = phi

            scattering_matrix = pd.concat([scattering_matrix, scattering_matrix_lower])
        scattering_matrix = scattering_matrix.query("theta >= 0.0")

        return dict(
            id=id,
            sphere_data=sphere_data,
            efficiencies=efficiencies,
            efficiencies_hem=efficiencies_hem,
            scattering_matrix=scattering_matrix,
        )

    def __read(self):
        output = None
        with open(self.output_file, "r") as fh:
            output = fh.read()

        if output is None:
            raise Exception("Could not read file")

        section_divider = "****************************************************\n"
        sections = output.split(section_divider)
        sections = [
            s
            for s in sections
            if ("calculation results for run" in s) or ("input variables for run " in s)
        ]

        inputs = []
        results = []
        for s in sections:
            if "input variables for run " in s:
                inputs.append(MSTM4Manager.__parse_input(s))
            elif "calculation results for run" in s:
                results.append(self.__parse_output(s))
        if (not results) or (not inputs):
            raise Exception("No results found in the file!")
        number_of_particles = (
            results[0]["sphere_data"].shape[0] if self.print_sphere_data else None
        )
        number_of_angles = results[0]["scattering_matrix"].shape[0]
        number_of_runs = len(inputs)
        sphere_data_header = (
            list(results[0]["sphere_data"]) if self.print_sphere_data else []
        )
        scattering_matrix_header = list(results[0]["scattering_matrix"])

        mstm = dict(
            length_scale_factor=np.zeros(number_of_runs),
            sphere_data={
                key: np.zeros((number_of_particles, number_of_runs))
                for key in sphere_data_header
            }
            if number_of_particles is not None
            else {},
            efficiencies=dict(
                q_ext_unp=np.zeros(number_of_runs),
                q_abs_unp=np.zeros(number_of_runs),
                q_sca_unp=np.zeros(number_of_runs),
                q_ext_par=np.zeros(number_of_runs),
                q_abs_par=np.zeros(number_of_runs),
                q_sca_par=np.zeros(number_of_runs),
                q_ext_per=np.zeros(number_of_runs),
                q_abs_per=np.zeros(number_of_runs),
                q_sca_per=np.zeros(number_of_runs),
            ),
            efficiencies_hem=dict(
                q_hem_u_unp=np.zeros(number_of_runs),
                q_hem_d_unp=np.zeros(number_of_runs),
                q_hem_u_par=np.zeros(number_of_runs),
                q_hem_d_par=np.zeros(number_of_runs),
                q_hem_u_per=np.zeros(number_of_runs),
                q_hem_d_per=np.zeros(number_of_runs),
            ),
            scattering_matrix={
                key: np.zeros((number_of_angles, number_of_runs))
                if key != "theta"
                else np.zeros((number_of_angles))
                for key in scattering_matrix_header
            },
        )

        for i, r in zip(inputs, results, strict=True):
            # inputs
            idx = i["id"] - 1
            mstm["length_scale_factor"][idx] = i["length_scale_factor"]

            # results
            idx = r["id"] - 1
            for key in sphere_data_header:
                mstm["sphere_data"][key][:, idx] = r["sphere_data"][key].to_numpy()
            for efficiency_type, value in r["efficiencies"].items():
                mstm["efficiencies"][efficiency_type][idx] = value
            for efficiency_type, value in r["efficiencies_hem"].items():
                mstm["efficiencies_hem"][efficiency_type][idx] = value
            for key in scattering_matrix_header:
                item = r["scattering_matrix"][key].to_numpy()
                if key == "theta":
                    mstm["scattering_matrix"][key] = item
                else:
                    mstm["scattering_matrix"][key][:, idx] = item

        self.output = mstm

    def clean(self):
        os.system(f"rm -f {self.input_file} {self.output_file} temp_pos.dat")

    def run(self, runner: pyperf.Runner | None = None, cleanup: bool = True):
        self.__write()
        self.__exec(runner)
        if runner is None:
            self.__read()

        if cleanup:
            self.clean()

    def export(self, cleanup: bool = False):
        if self.config is None:
            raise Exception("Config isn't set")
        elif self.config.output_filename is None:
            raise Exception("No output filename provided")
        file = self.config.output_filename
        file = file.with_name("mstm_" + file.name)

        if not self.output:
            self.run(cleanup=cleanup)
        mstm = copy.deepcopy(self.output)

        mstm["length_scale_factor"] = mstm["length_scale_factor"].tolist()
        for tag in [
            "sphere_data",
            "efficiencies",
            "efficiencies_hem",
            "scattering_matrix",
        ]:
            for key, value in mstm[tag].items():
                mstm[tag][key] = value.tolist()

        radius_equivalent = np.sum(self.config.spheres[:, 3] ** 3) ** 1 / 3
        geometric_cross_section = np.pi * radius_equivalent**2
        extinction_cross_section = (
            self.output["efficiencies"]["q_ext_unp"] * geometric_cross_section
        )
        scattering_cross_section = (
            self.output["efficiencies"]["q_sca_unp"] * geometric_cross_section
        )
        exporter = Export(
            source="mstm4",
            scale=self.config.wavelength_scale,
            particles=dict(
                position=self.config.spheres[:, :3].tolist(),
                radii=self.config.spheres[:, 3].tolist(),
                refractive_index=self._ref_idx.tolist(),
                radius_of_gyration=radius_of_gyration(
                    self.config.spheres[:, :3],
                    self.config.spheres[:, 3],
                ),
                metadata=self.config.particle_metadata,
            ),
            wavelength=dict(
                value=self.config.wavelength.tolist(),
                geometric_cross_section=geometric_cross_section,
                extinction_cross_section=extinction_cross_section,
                scattering_cross_section=scattering_cross_section,
                extinction_efficiency=self.output["efficiencies"]["q_ext_unp"].tolist(),
                scattering_efficiency=self.output["efficiencies"]["q_sca_unp"].tolist(),
                single_scattering_albedo=(
                    self.output["efficiencies"]["q_sca_unp"]
                    / self.output["efficiencies"]["q_ext_unp"]
                ).tolist(),
                medium_refractive_index=self.config.medium_refractive_index.tolist(),
            ),
            angle=dict(
                theta=self.output["scattering_matrix"]["theta"].tolist(),
                phase_function=self.output["scattering_matrix"]["11"].tolist(),
                degree_of_linear_polarization=(
                    -self.output["scattering_matrix"]["12"]
                ).tolist(),
                # degree_of_linear_polarization=(
                #     -self.output["scattering_matrix"]["12"]
                #     / self.output["scattering_matrix"]["11"]
                # ).tolist(),
                # degree_of_linear_polarization_q=1,
                # degree_of_linear_polarization_u=1,
                degree_of_circular_polarization=(
                    self.output["scattering_matrix"]["41"]
                ).tolist()
                if "41" in self.output["scattering_matrix"]
                else [],
                depolarization_ratio=(
                    1 - self.output["scattering_matrix"]["22"]
                ).tolist(),
                anisotropy_parameter=(
                    self.output["scattering_matrix"]["33"]
                    - self.output["scattering_matrix"]["44"]
                ).tolist(),
            ),
        )
        exporter.save(filename=file)

        if cleanup:
            self.clean()


if __name__ == "__main__":
    mstm = MSTM4Manager(
        path_config=str(project_root / "examples/graphite/config.json"),
        parallel=4,
    )
    # See params at
    # https://pyperf.readthedocs.io/en/latest/api.html#runner-class
    runner = pyperf.Runner(values=1, processes=5, warmups=1)
    mstm.run(runner)
    # mstm.export("mstm4.bz2", True)
