"""Convenience wrapper around the MSTM3 binary.

Provides a small manager used in benchmarks and regression comparisons.
"""

import os
import re
import bz2
import json
import yaml
import copy
import pyperf
import _pickle
from io import StringIO

import numpy as np
import pandas as pd

from yasfpy.config import Config


class MSTM3Manager:
    """Manage MSTM3 benchmark runs.

    The manager can generate an MSTM3 input file from a YASF configuration, run
    the external MSTM3 binary, and parse the resulting output file into basic
    scattering quantities.

    Notes
    -----
    This helper is intended for benchmarking/regression comparisons rather than
    production simulations.
    """

    binary: str = "./mstm"
    input_file: str = "mstm3.inp"
    output_file: str = "mstm3.dat"

    def __init__(
        self,
        config_path: str | None = None,
        binary: str | None = None,
        input_file: str = "mstm3.inp",
        output_file: str = "mstm3.dat",
    ):
        """Initialize the MSTM3 manager.

        Parameters
        ----------
        config_path:
            Path to a YASF configuration file.
        binary:
            Path to the MSTM3 executable.
        input_file:
            Filename for the generated MSTM3 input.
        output_file:
            Filename for the MSTM3 output.
        """
        self.config = Config(config_path)
        self.binary = binary

        self.input_file = input_file
        self.output_file = output_file

    def __write(
        self,
        fixed_lmax: bool = True,
        save: bool = True,
    ) -> None:
        mstm_config = "output_file\n"
        mstm_config += f"{self.output_file}\n"
        mstm_config += "write_sphere_data\n"
        mstm_config += "1\n"
        mstm_config += "calculate_scattering_coefficients\n"
        mstm_config += "1\n"
        # mstm_config += "scattering_coefficient_file\n"
        # mstm_config += "mstm3.coef\n"
        mstm_config += "incident_or_target_frame\n"
        mstm_config += "0\n"
        mstm_config += "min_scattering_angle_deg\n"
        mstm_config += "0.0d0\n"
        mstm_config += "max_scattering_angle_deg\n"
        mstm_config += "180.d0\n"
        mstm_config += "min_scattering_plane_angle_deg\n"
        mstm_config += "0.0d0\n"
        mstm_config += "max_scattering_plane_angle_deg\n"
        mstm_config += "360.0d0\n"
        mstm_config += "delta_scattering_angle_deg\n"
        mstm_config += "30\n"
        mstm_config += "normalize_scattering_matrix\n"
        mstm_config += "0\n"
        mstm_config += "incident_polar_angle_deg\n"
        mstm_config += f"{self.config.config['initial_field']['polar_angle']:e}\n"
        mstm_config += "incident_azimuth_angle_deg\n"
        mstm_config += f"{self.config.config['initial_field']['azimuthal_angle']:e}\n"
        mstm_config += "real_ref_index_scale_factor\n"
        mstm_config += "1.0d0\n"
        mstm_config += "imag_ref_index_scale_factor\n"
        mstm_config += "1.0d0\n"
        mstm_config += "real_chiral_factor\n"
        mstm_config += "0.0d0\n"
        mstm_config += "imag_chiral_factor\n"
        mstm_config += "0.0d0\n"
        mstm_config += "medium_real_chiral_factor\n"
        mstm_config += "0.d0\n"
        mstm_config += "medium_imag_chiral_factor\n"
        mstm_config += "0.d0\n"

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
            mstm_config += f"{2 * np.pi / (wl * self.config.wavelength_scale / self.config.particles_scale):e}\n"
            mstm_config += "medium_real_ref_index\n"
            mstm_config += f"{self.config.medium_refractive_index[wl_idx].real}\n"
            mstm_config += "medium_imag_ref_index\n"
            mstm_config += f"{self.config.medium_refractive_index[wl_idx].imag}\n"
            mstm_config += "number_spheres\n"
            mstm_config += f"{self.config.spheres.shape[0]}\n"

            mstm_config += "sphere_sizes_and_positions\n"
            for particle_idx in range(self.config.spheres.shape[0]):
                position_str = ",".join(
                    [str(i) for i in self.config.spheres[particle_idx, [3, 0, 1, 2]]]
                )
                ref_idx_str = f"{ref_idx[particle_idx, wl_idx].real}, {ref_idx[particle_idx, wl_idx].imag}"
                mstm_config += f"{position_str},{ref_idx_str}\n"

            if wl_idx + 1 == self.config.wavelength.size:
                mstm_config += "end_of_options"
            else:
                mstm_config += "new_run\n"

        if save:
            with open(self.input_file, "w") as fh:
                fh.write(mstm_config)

        return mstm_config

    def __exec(self, runner: pyperf.Runner | None = None):
        command = [self.binary, self.input_file]
        if runner is None:
            os.system(" ".join(command))
        else:
            runner.bench_command("mstm3_exec", command)

    @staticmethod
    def __parse_output(results):
        p = re.compile(r"input parameters for run number\s+(\d+)")
        r = p.search(results)
        id = r.group(1)
        id = int(id)

        p = re.compile(r"length, ref. indx. scale factors:\s+(.+?)\s+(.+?)\s+(.+?)")
        r = p.search(results)
        length_scale_factor = r.group(1)
        length_scale_factor = float(length_scale_factor)

        p = re.compile(
            r"sphere host\s+ka\s+x-x\(host\)\s+y-y\(host\)\s+z-z\(host\)\s+Re\(m\)\s+Im\(m\)\s+ Qext\s+Qsca\s+Qabs\s+Qabs\(V\)(.+)unpolarized",
            re.S,
        )
        r = p.search(results)
        sphere_data = r.group(1)
        sphere_data = pd.read_csv(
            StringIO(sphere_data),
            names=[
                "sphere host",
                "ka",
                "x-x(host)",
                "y-y(host)",
                "z-z(host)",
                "Re(m)",
                "Im(m)",
                "Qext",
                "Qsca",
                "Qabs",
                "Qabs(V)",
            ],
            header=None,
            sep=r"\s+",
        )

        efficiencies = {}

        p = re.compile(
            r"unpolarized total ext, abs, scat efficiencies, w.r.t. xv, and asym. parm\s+(.+)"
        )
        r = p.search(results)
        efficiencies_unp = r.group(1)
        efficiencies_unp = [float(e) for e in efficiencies_unp.split()]
        efficiencies["q_ext_unp"] = efficiencies_unp[0]
        efficiencies["q_abs_unp"] = efficiencies_unp[1]
        efficiencies["q_sca_unp"] = efficiencies_unp[2]

        p = re.compile(r"parallel total ext, abs, scat efficiencies\s+(.+)")
        r = p.search(results)
        efficiencies_par = r.group(1)
        efficiencies_par = [float(e) for e in efficiencies_par.split()]
        efficiencies["q_ext_par"] = efficiencies_par[0]
        efficiencies["q_abs_par"] = efficiencies_par[1]
        efficiencies["q_sca_par"] = efficiencies_par[2]

        p = re.compile(r"perpendicular total ext, abs, scat efficiencies\s+(.+)")
        r = p.search(results)
        efficiencies_per = r.group(1)
        efficiencies_per = [float(e) for e in efficiencies_per.split()]
        efficiencies["q_ext_per"] = efficiencies_per[0]
        efficiencies["q_abs_per"] = efficiencies_per[1]
        efficiencies["q_sca_per"] = efficiencies_per[2]

        p = re.compile(
            r"theta\s+phi\s+11\s+12\s+13\s+14\s+21\s+22\s+23\s+24\s+31\s+32\s+33\s+34\s+41\s+42\s+43\s+44\s+(.+)",
            re.S,
        )
        r = p.search(results)
        scattering_matrix = r.group(1)
        scattering_matrix = pd.read_csv(
            StringIO(scattering_matrix),
            names=[
                "theta",
                "phi",
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
            sep=r"\s+",
        )

        return dict(
            id=id,
            length_scale_factor=length_scale_factor,
            sphere_data=sphere_data,
            efficiencies=efficiencies,
            scattering_matrix=scattering_matrix,
        )

    def __read(self):
        output = None
        with open(self.output_file, "r") as fh:
            output = fh.read()

        if output is None:
            raise Exception("Could not read file")

        section_divider = "*****************************************************\n"
        sections = output.split(section_divider)

        results = []
        for s in sections:
            results.append(MSTM3Manager.__parse_output(s))
        if not results:
            raise Exception("No results found in the file!")
        number_of_particles = results[0]["sphere_data"].shape[0]
        number_of_angles = results[0]["scattering_matrix"].shape[0]
        number_of_runs = len(results)
        sphere_data_header = list(results[0]["sphere_data"])
        scattering_matrix_header = list(results[0]["scattering_matrix"])

        mstm = dict(
            length_scale_factor=np.zeros(number_of_runs),
            # sphere_data=[None] * number_of_runs,
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
            # scattering_matrix=[None] * number_of_runs,
            scattering_matrix={
                key: np.zeros((number_of_angles, number_of_runs))
                for key in scattering_matrix_header
            },
        )
        for r in results:
            idx = r["id"] - 1
            mstm["length_scale_factor"][idx] = r["length_scale_factor"]
            for key in sphere_data_header:
                mstm["sphere_data"][key][:, idx] = r["sphere_data"][key].to_numpy()
            for efficiency_type, value in r["efficiencies"].items():
                mstm["efficiencies"][efficiency_type][idx] = value
            for key in scattering_matrix_header:
                mstm["scattering_matrix"][key][:, idx] = r["scattering_matrix"][
                    key
                ].to_numpy()

        self.output = mstm

    def clean(self):
        """Delete generated MSTM3 files."""
        os.system(f"rm -f {self.input_file} {self.output_file} mstm3.coef")

    def run(self, runner: pyperf.Runner | None = None, cleanup: bool = True):
        """Run MSTM3 and parse output.

        Parameters
        ----------
        runner:
            Optional :class:`pyperf.Runner` to benchmark the command. When omitted,
            the binary is executed directly and outputs are parsed.
        cleanup:
            If True, remove generated input/output files afterwards.
        """
        self.__write()
        self.__exec(runner)
        if runner is None:
            self.__read()

        if cleanup:
            self.clean()

    def export(self, file: str | None = None, cleanup: bool = False):
        """Run MSTM3 and export parsed results.

        Parameters
        ----------
        file:
            Output filename. Supported extensions are ``.json``, ``.yaml``/``.yml``,
            and ``.bz2``.
        cleanup:
            If True, also remove generated MSTM3 files afterwards.
        """
        if file is None:
            raise Exception("Please provide a filename")
        self.run(cleanup=False)
        mstm = copy.deepcopy(self.output)

        mstm["length_scale_factor"] = mstm["length_scale_factor"].tolist()
        for tag in ["sphere_data", "efficiencies", "scattering_matrix"]:
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
    mstm = MSTM3Manager("./config.json", "./MSTM3/src/mstm")
    # See params at
    # https://pyperf.readthedocs.io/en/latest/api.html#runner-class
    runner = pyperf.Runner(values=1, processes=5, warmups=1)
    mstm.run(runner)
    # mstm.export("mstm3.bz2", True)
