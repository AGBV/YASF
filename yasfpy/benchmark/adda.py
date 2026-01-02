"""Wrapper for ADDA-based benchmarks.

This module provides a manager/helper to run ADDA (if available) and collect
results for comparison/benchmarking.
"""

import os, json, yaml, copy, bz2, re, shutil
import _pickle
import pyperf
from yasfpy.config import Config

DEFAULT_BIN_PATH = ""
DEFAULT_CONF_PATH = ""


class ADDAManager:
    """Manage and run ADDA benchmarks.

    This is a lightweight wrapper that:

    - reads a YASF configuration via :class:`yasfpy.config.Config`
    - runs the external ADDA binary (if present)
    - parses basic cross sections/efficiencies from ADDA output

    Notes
    -----
    This module is primarily used for benchmarking and comparisons; it does not
    attempt to expose the full ADDA feature set.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONF_PATH,
        binary: str = DEFAULT_BIN_PATH,
        output_file: str = "wow",
    ):
        """Initialize the ADDA manager.

        Parameters
        ----------
        config_path:
            Path to a YASF configuration file.
        binary:
            Path to the ADDA executable.
        output_file:
            Identifier used for exported results.
        """
        self.config = Config(config_path)
        self.binary = binary
        self.output_file = output_file

    def __exec(self, runner: pyperf.Runner | None = None):
        # command = [self.binary, self.input_args]
        if self.config.spheres.shape[0] > 1:
            print(
                "WARNING! MORE THAN 1 SPHERE DETECTED IN CONFIG! USING ONLY THE FIRST SPHERE!"
            )

        diameter = (
            self.config.spheres[0, 3] * 2
        )  # TODO: THIS ALWAYS NEEDS TO BE IN NANOMETERS
        for wi, wavelength in enumerate(self.config.wavelength):
            command = [
                self.binary,
                f"-size",
                f"{diameter}",
                f"-lambda",
                f"{wavelength}",
                f"-m",
                f"{self.config.medium_refractive_index[wi].real}",
                f"{self.config.medium_refractive_index[wi].imag}",
                "-asym",
                f"-dir",
                f"run_sphere_r{diameter}_l{wavelength}",
            ]
        if runner is None:
            os.system(" ".join(command))
        else:
            # TODO: this needs to take into account all wavelengths
            runner.bench_command("mstm3_exec", command)

    def __read(self):
        q_exts: list[float] = []
        q_scats: list[float] = []
        gs: list[float] = []
        c_exts: list[float] = []
        c_scats: list[float] = []
        diameter = self.config.spheres[0, 3] * 2
        for wavelength in self.config.wavelength:
            with open(f"run_sphere_r{diameter}_l{wavelength}/CrossSec-Y", "r") as f:
                data = f.readlines()
            for line in data:
                if len(re.findall("^g\t=", line)) > 0:
                    g = float(line.split("=")[-1].strip().split(",")[-1].strip(")"))
                    gs.append(g)
                if len(re.findall("Qsca\t", line)) > 0:
                    q_sca = float(line.split("=")[-1].strip())
                    q_scats.append(q_sca)
                if len(re.findall("Qext\t", line)) > 0:
                    q_ext = float(line.split("=")[-1].strip())
                    q_exts.append(q_ext)
                if len(re.findall("Csca\t", line)) > 0:
                    c_sca = float(line.split("=")[-1].strip())
                    c_scats.append(c_sca)
                if len(re.findall("Cext\t", line)) > 0:
                    c_ext = float(line.split("=")[-1].strip())
                    c_exts.append(c_ext)
        adda = {
            "cross_sections": {"C_scat": c_scats, "C_ext": c_ext},
            "efficiencies": {
                "Q_scat": q_scats,
                "Q_ext": q_exts,
            },
            "asymmetry": {"g": gs},
        }
        self.output = adda

    def clean(self):
        """Remove ADDA run directories.

        Notes
        -----
        This deletes directories matching ``run_sphere*`` in the current working
        directory.
        """
        # TODO: clean folders
        res_dirs = [i for i in os.listdir() if len(re.findall("run_sphere", i)) > 0]
        for dir in res_dirs:
            shutil.rmtree(dir)
        pass

    def run(self, runner: pyperf.Runner | None = None, cleanup: bool = True):
        """Run ADDA and optionally parse output.

        Parameters
        ----------
        runner:
            Optional :class:`pyperf.Runner` to benchmark the command. When omitted,
            the external binary is executed directly and outputs are parsed.
        cleanup:
            If True, delete ADDA output directories after the run.
        """
        self.__exec(runner)
        if runner is None:
            self.__read()

        if cleanup:
            self.clean()

    def export(self, file: str | None = None, cleanup: bool = False):
        """Run ADDA and export results to a file.

        Parameters
        ----------
        file:
            Output filename. Supported extensions are ``.json``, ``.yaml``/``.yml``,
            and ``.bz2``.
        cleanup:
            If True, delete ADDA output directories after exporting.
        """
        if file is None:
            raise Exception("Please provide a filename")
        self.run(cleanup=False)
        adda = copy.deepcopy(self.output)

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
    adda = ADDAManager("./config.json", "./MSTM3/src/mstm")
    # See params at
    # https://pyperf.readthedocs.io/en/latest/api.html#runner-class
    runner = pyperf.Runner(values=1, processes=5, warmups=1)
    adda.run(runner)
    # adda.export("adda.bz2", True)
