# Usage

## Command Line Interface

The package installs a `yasf` CLI.

```sh
yasf --help
yasf compute --help
```

The main entry point is:

```sh
yasf compute --config path/to/config.json
```

Selected options (see `--help` for the full list):

- `--cluster PATH`: override `particles.geometry.file`
- `--cluster-scale FLOAT`: override `particles.geometry.scale`
- `--cluster-dimensional-scale FLOAT`: override `particles.geometry.dimensional_scale`
- `--backend {yasf,mstm}`: use the native YASF backend or run the MSTM4 comparison backend

## Cluster Scaling Semantics

YASF has two different *multiplicative* scaling concepts for particle clusters:

### `particles.geometry.scale` (unit conversion / wavelength-unit matching)

This value is a **unit scale factor** for the positions/radii loaded from the cluster file.
It is used together with the wavelength scale to ensure geometry and wavelength are expressed in compatible units.

- Geometry values are assumed to be in “cluster units” and are converted to the same unit system used by `parameters.wavelength.data`.
- In practice, if your wavelength is specified in micrometers (`scale = 1e-6`), then your cluster scale should also typically be `1e-6` when your cluster file values are also in micrometers.

### `particles.geometry.dimensional_scale` (physical resizing)

This value is an additional **dimensionless** scaling factor applied to the entire cluster geometry:

- Multiplies `x`, `y`, `z`, and `r` by the same factor.
- Useful for “inflating/deflating” a cluster without rewriting the geometry file.

### Order of operations

When loading a config, `yasfpy.config.Config` applies scaling in the following order:

1. Load raw geometry from the cluster file (x, y, z, r)
2. Apply **dimensional scaling** (`dimensional_scale` / `--cluster-dimensional-scale`)
3. Apply overlap correction (if enabled by the code path)
4. Convert geometry into wavelength units via the ratio:
   `particles.geometry.scale / parameters.wavelength.scale`

### Precedence (config vs CLI/API)

- If you set CLI overrides (`yasf compute --cluster-scale ...` / `--cluster-dimensional-scale ...`), they take precedence over the config file.
- If you don’t pass overrides (or you call `YASF(...)` without them), the values from the config file are used.

## Python API

The Python API is outlined in the [API documentation](api.md).

You can also override scaling programmatically:

```py
from yasfpy import YASF

sim = YASF(
    path_config="config.json",
    cluster_scale=1e-6,
    cluster_dimensional_scale=1.5,
)
sim.run()
```

Please also check the [examples](examples.md) section for runnable examples in this repository.
