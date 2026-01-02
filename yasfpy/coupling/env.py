"""Environment-variable helpers for coupling backends.

These helpers centralize parsing/normalization of environment variables used to
control coupling backend behavior (GPU strategy selection, chunk sizes,
experimental toggles).

Notes
-----
These are intentionally forgiving: invalid inputs fall back to defaults rather
than raising, to keep CLI and batch runs robust.
"""

from __future__ import annotations

import os


def parse_bool_env(name: str, *, default: bool) -> bool:
    """Parse a boolean environment variable.

    Parameters
    ----------
    name:
        Environment variable name.
    default:
        Default value used when the variable is unset or invalid.

    Returns
    -------
    bool
        Parsed boolean value.
    """

    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return default if raw == "" else False
    return default


def parse_int_env(name: str, *, default: int, minimum: int = 1) -> int:
    """Parse an integer environment variable with a lower bound.

    Parameters
    ----------
    name:
        Environment variable name.
    default:
        Default value used when the variable is unset or invalid.
    minimum:
        Lower bound enforced on the returned value.

    Returns
    -------
    int
        Parsed integer value (at least ``minimum``).
    """

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def normalize_gpu_strategy(value: str) -> str:
    """Normalize the GPU coupling strategy selector.

    Parameters
    ----------
    value:
        A raw environment variable value.

    Returns
    -------
    str
        One of ``{'auto', 'thread_row', 'block_row'}``.
    """

    value = value.strip()
    if value in {"auto", "thread_row", "block_row"}:
        return value
    return "auto"


def normalize_multi_wavelength_mode(value: str) -> str:
    """Normalize the multi-wavelength GPU execution mode selector.

    Parameters
    ----------
    value:
        A raw environment variable value.

    Returns
    -------
    str
        One of ``{'auto', 'atomic', 'per_wavelength', 'per_wavelength_chunked'}``.
    """

    value = value.strip()
    if value in {"auto", "atomic", "per_wavelength", "per_wavelength_chunked"}:
        return value
    return "auto"
