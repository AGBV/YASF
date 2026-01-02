import numpy as np

from yasfpy.fmm import HelmholtzTreecode


def _direct_sum(k: complex, pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        acc = 0.0 + 0.0j
        for j in range(n):
            if i == j:
                continue
            d = pos[i] - pos[j]
            r = float(np.sqrt(np.dot(d, d)))
            acc += np.exp(1j * k * r) / r * q[j]
        out[i] = acc
    return out


def test_helmholtz_treecode_matches_direct_sum_reasonably():
    rng = np.random.default_rng(0)
    n = 200
    pos = rng.normal(size=(n, 3))
    pos *= 0.8

    q = rng.normal(size=n) + 1j * rng.normal(size=n)
    k = 2.5 + 0.0j

    ref = _direct_sum(k, pos, q)

    tc = HelmholtzTreecode(k=k, order=4, leaf_size=24, theta=0.6)
    tc.build(pos)
    approx = tc.apply(q)

    rel_err = np.linalg.norm(approx - ref) / np.linalg.norm(ref)
    assert rel_err < 5e-2
