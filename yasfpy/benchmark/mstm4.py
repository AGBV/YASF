# import os
# os.system("python mstm4-write.py")
# os.system("./MSTM/code/mstm ./mstm4.inp")
# os.system("python mstm4-read.py")

import bz2
import copy
import json
import os
import re
from io import StringIO

import _pickle
import numpy as np
import pandas as pd
import pyperf
import yaml

from yasfpy.config import Config


class MSTM4Manager:
    binary: str = "./mstm"
    input_file: str = "mstm4.inp"
    output_file: str = "mstm4.dat"
    parallel: int = 1

    def __init__(
        self,
        config_path: str = None,
        binary: str = None,
        input_file: str = "mstm4.inp",
        output_file: str = "mstm4.dat",
        parallel: int = 1,
    ):
        self.config = Config(config_path)
        self.binary = binary

        self.input_file = input_file
        self.output_file = output_file

        self.parallel = parallel

    def __write(
        self,
        fixed_lmax: bool = True,
        save: bool = True,
    ) -> None:
        mstm_config = "output_file\n"
        mstm_config += f"{self.output_file}\n"
        mstm_config += "print_sphere_data\n"
        mstm_config += "t\n"
        mstm_config += "calculate_scattering_matrix\n"
        mstm_config += "t\n"
        mstm_config += "incident_frame\n"
        mstm_config += "f\n"
        mstm_config += "scattering_map_model\n"
        mstm_config += "0\n"
        mstm_config += "scattering_map_increment\n"
        mstm_config += "30.d0\n"
        mstm_config += "incident_beta_deg\n"
        mstm_config += f"{self.config.config['initial_field']['polar_angle']:e}\n"
        mstm_config += "incident_alpha_deg\n"
        mstm_config += f"{self.config.config['initial_field']['azimuthal_angle']:e}\n"

        if fixed_lmax:
            mstm_config += "mie_epsilon\n"
            mstm_config += f"-{int(self.config.config['numerics']['lmax'])}\n"

        ref_idx = np.take(
            self.config.refractive_index_interpolated,
            self.config.spheres[:, 4].astype(int),
            axis=0,
        )
        for wl_idx, wl in enumerate(self.config.wavelength):
            mstm_config += "length_scale_factor\n"
            mstm_config += f"{2 * np.pi / (wl * self.config.wavelength_scale / self.config.particles_scale)}\n"
            mstm_config += "layer_ref_index\n"
            mstm_config += f"({self.config.medium_refractive_index[wl_idx].real}, {self.config.medium_refractive_index[wl_idx].imag})\n"
            mstm_config += "number_spheres\n"
            mstm_config += f"{self.config.spheres.shape[0]}\n"

            mstm_config += "sphere_data\n"
            for particle_idx in range(self.config.spheres.shape[0]):
                position_str = ",".join(
                    [str(i) for i in self.config.spheres[particle_idx, :-1]]
                )
                ref_idx_str = f"({ref_idx[particle_idx, wl_idx].real}, {ref_idx[particle_idx, wl_idx].imag})"
                mstm_config += f"{position_str},{ref_idx_str}\n"

            mstm_config += "end_of_sphere_data\n"

            if wl_idx + 1 == self.config.wavelength.size:
                mstm_config += "end_of_options"
            else:
                mstm_config += "new_run\n"

        if save:
            with open(self.input_file, "w") as fh:
                fh.write(mstm_config)

        return mstm_config

    # def __exec(self):
    #     os.system(f"{self.binary} {self.input_file}")

    def __exec(self, runner: pyperf.Runner = None):
        if self.parallel == 1:
            command = [self.binary, self.input_file]
        elif self.parallel % 4 == 0:
            command = [
                "mpiexec",
                "-n",
                str(self.parallel),
                self.binary,
                self.input_file,
            ]
        else:
            raise Exception("parallel parameter needs to be divisble by 4!")

        if runner is None:
            os.system(" ".join(command))
        else:
            runner.bench_command(f"mstm4_exec_{self.parallel}", command)

    @staticmethod
    def __parse_input(input):
        # Get id of run
        p = re.compile(r"\s+input variables for run\s+(\d+)")
        r = p.search(input)
        id = r.group(1)
        id = int(id)

        # lengths scale factor
        p = re.compile(r"\s+length, ref index scale factors\s+(.+?)\s+(.+?)\s+(.+?)")
        r = p.search(input)
        length_scale_factor = r.group(1)
        length_scale_factor = float(length_scale_factor)

        return dict(
            id=id,
            length_scale_factor=length_scale_factor,
        )

    @staticmethod
    def __parse_output(results):
        # Get id of run
        p = re.compile(r"\s+calculation results for run\s+(\d+)")
        r = p.search(results)
        id = r.group(1)
        id = int(id)

        p = re.compile(r"sphere\s+Qext\s+Qabs\s+Qvabs\s+(.+)total", re.S)
        r = p.search(results)
        sphere_data = r.group(1)
        sphere_data = pd.read_csv(
            StringIO(sphere_data),
            names=["sphere", "Qext", "Qabs", "Qvabs"],
            header=None,
            sep="\s+",
        )

        p = re.compile(
            r"total extinction, absorption, scattering efficiencies \(unpol, par, perp incidence\)\s+(.+)"
        )
        r = p.search(results)
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

        p = re.compile(
            r"down and up hemispherical scattering efficiencies \(unpol, par, perp\)\s+(.+)"
        )
        r = p.search(results)
        efficiencies_hem = r.group(1)
        efficiencies_hem = [float(e) for e in efficiencies_hem.split()]
        efficiencies_hem = dict(
            q_hem_u_unp=efficiencies_hem[0],
            q_hem_d_unp=efficiencies_hem[1],
            q_hem_u_par=efficiencies_hem[2],
            q_hem_d_par=efficiencies_hem[3],
            q_hem_u_per=efficiencies_hem[4],
            q_hem_d_per=efficiencies_hem[5],
        )

        p = re.compile(
            r"theta\s+11\s+12\s+13\s+14\s+21\s+22\s+23\s+24\s+31\s+32\s+33\s+34\s+41\s+42\s+43\s+44\s+(.+)",
            re.S,
        )
        r = p.search(results)
        scattering_matrix = r.group(1)
        scattering_matrix = pd.read_csv(
            StringIO(scattering_matrix),
            names=[
                "theta",
                "11",
                "12",
                "13",
                "14",
                "21",
                "22",
                "23",
                "24",
                "31",
                "32",
                "33",
                "34",
                "41",
                "42",
                "43",
                "44",
            ],
            header=None,
            sep="\s+",
        )

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
                results.append(MSTM4Manager.__parse_output(s))
        if (not results) or (not inputs):
            raise Exception("No results found in the file!")
        number_of_particles = results[0]["sphere_data"].shape[0]
        number_of_angles = results[0]["scattering_matrix"].shape[0]
        number_of_runs = len(inputs)
        sphere_data_header = list(results[0]["sphere_data"])
        scattering_matrix_header = list(results[0]["scattering_matrix"])

        mstm = dict(
            length_scale_factor=np.zeros(number_of_runs),
            sphere_data={
                key: np.zeros((number_of_particles, number_of_runs))
                for key in sphere_data_header
            },
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
                for key in scattering_matrix_header
            },
        )

        for i, r in zip(inputs, results):
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
                mstm["scattering_matrix"][key][:, idx] = r["scattering_matrix"][
                    key
                ].to_numpy()

        self.output = mstm

    def clean(self):
        os.system(f"rm -f {self.input_file} {self.output_file} temp_pos.dat")

    def run(self, runner: pyperf.Runner = None, cleanup: bool = True):
        self.__write()
        self.__exec(runner)
        if runner is None:
            self.__read()

        if cleanup:
            self.clean()

    def export(self, file: str = None, cleanup: bool = False):
        if file is None:
            raise Exception("Please provide a filename")
        self.run(cleanup=False)
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

        match file.split(".")[-1]:
            case "json":
                with open(file, "w") as outfile:
                    json.dump(self.output, outfile)
            case "yaml" | "yml":
                with open(file, "w") as outfile:
                    yaml.dump(self.output, outfile, default_flow_style=False)
            case "bz2":
                with bz2.BZ2File(file, "w") as outfile:
                    _pickle.dump(self.output, outfile)
            case _:
                raise Exception(
                    "The provided config file needs to be a json or yaml file!"
                )

        if cleanup:
            self.clean()


if __name__ == "__main__":
    mstm = MSTM4Manager("./config.json", "./MSTM/code/mstm", parallel=4)
    # See params at
    # https://pyperf.readthedocs.io/en/latest/api.html#runner-class
    runner = pyperf.Runner(values=1, processes=5, warmups=1)
    mstm.run(runner)
    # mstm.export("mstm4.bz2", True)
