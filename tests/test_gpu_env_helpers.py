import os

import pytest

from yasfpy.coupling.env import (
    normalize_gpu_strategy as _normalize_gpu_strategy,
    normalize_multi_wavelength_mode as _normalize_multi_wavelength_mode,
    parse_bool_env as _parse_bool_env,
    parse_int_env as _parse_int_env,
)


def test_parse_bool_env_defaults():
    os.environ.pop("YASF_TEST_BOOL", None)
    assert _parse_bool_env("YASF_TEST_BOOL", default=True) is True
    assert _parse_bool_env("YASF_TEST_BOOL", default=False) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("unexpected", True),
    ],
)
def test_parse_bool_env_values(raw: str, expected: bool):
    os.environ["YASF_TEST_BOOL"] = raw
    assert _parse_bool_env("YASF_TEST_BOOL", default=True) is expected


def test_parse_int_env_defaults_and_minimum():
    os.environ.pop("YASF_TEST_INT", None)
    assert _parse_int_env("YASF_TEST_INT", default=7, minimum=3) == 7

    os.environ["YASF_TEST_INT"] = ""
    assert _parse_int_env("YASF_TEST_INT", default=7, minimum=3) == 7

    os.environ["YASF_TEST_INT"] = "2"
    assert _parse_int_env("YASF_TEST_INT", default=7, minimum=3) == 3

    os.environ["YASF_TEST_INT"] = "10"
    assert _parse_int_env("YASF_TEST_INT", default=7, minimum=3) == 10


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("block_row", "block_row"),
        ("thread_row", "thread_row"),
        ("auto", "auto"),
        ("", "auto"),
        ("garbage", "auto"),
    ],
)
def test_normalize_gpu_strategy(raw: str, expected: str):
    assert _normalize_gpu_strategy(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("atomic", "atomic"),
        ("per_wavelength", "per_wavelength"),
        ("per_wavelength_chunked", "per_wavelength_chunked"),
        ("auto", "auto"),
        ("", "auto"),
        ("garbage", "auto"),
    ],
)
def test_normalize_multi_wavelength_mode(raw: str, expected: str):
    assert _normalize_multi_wavelength_mode(raw) == expected
