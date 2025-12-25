import numpy as np
import numpy.testing as npt

from yasfpy.mueller import jones_to_mueller, jones_to_mueller_numba


def test_jones_to_mueller_numba_matches_reference_batched():
    rng = np.random.default_rng(0)

    # Test a non-trivial batch shape and include non-contiguous input.
    j = rng.normal(size=(3, 5, 2, 2)) + 1j * rng.normal(size=(3, 5, 2, 2))
    j = j[:, ::2, :, :]  # non-contiguous view

    ref = jones_to_mueller(j)
    fast = jones_to_mueller_numba(j)

    assert ref.shape == fast.shape == j.shape[:-2] + (4, 4)
    assert np.isrealobj(ref)
    assert np.isrealobj(fast)

    npt.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)
