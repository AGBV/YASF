"""Initial (incident) field coefficient generation.

This module contains the implementation that used to live in
`yasfpy.simulation.Simulation.compute_initial_field_coefficients`, extracted to
keep `yasfpy/simulation.py` smaller.

YASF represents incident and scattered fields using a vector-spherical-
wave-function (VSWF) expansion about each particle center. This module computes
VSWF expansion coefficients for common incident fields (currently plane waves
and a normal-incidence Gaussian-wavebundle variant).

References
----------
VSWF expansions for plane-wave illumination are standard and can be found in
many texts, e.g. :cite:`Mishchenko-2002-ID6`.

The wavebundle-style Gaussian beam construction is adapted from CELES-style
implementations :cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from yasfpy.functions.misc import multi2single_index, transformation_coefficients
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


def compute_initial_field_coefficients(sim: "Simulation") -> None:
    r"""Compute incident-field VSWF expansion coefficients.

    Parameters
    ----------
    sim:
        Simulation instance providing incident-field parameters
        (``sim.parameters.initial_field``), particle positions, and numerical
        truncation (``sim.numerics.lmax``).

    Returns
    -------
    None

    Notes
    -----
    The computed coefficients are written to ``sim.initial_field_coefficients``
    with shape ``(particles, nmax, channels)``.

    Currently supported cases:

    - Plane wave (default when ``beam_width`` is ``0`` or infinite)
    - Normal-incidence Gaussian wavebundle (finite positive ``beam_width`` and
      ``normal_incidence=True``)

    Unsupported configurations log an error.
    """

    sim.log.info("compute initial field coefficients ...")

    if np.isfinite(sim.parameters.initial_field.beam_width) and (
        sim.parameters.initial_field.beam_width > 0
    ):
        sim.log.info("\t Gaussian beam ...")
        if sim.parameters.initial_field.normal_incidence:
            _compute_wavebundle_normal_incidence(sim)
        else:
            sim.log.error("\t this case is not implemented")
    else:
        sim.log.info("\t plane wave ...")
        _compute_planewave(sim)

    sim.log.info("done")


def _compute_planewave(sim: "Simulation") -> None:
    r"""Compute plane-wave incident-field coefficients.

    Parameters
    ----------
    sim:
        Simulation instance. The result is stored on
        ``sim.initial_field_coefficients``.

    Notes
    -----
    Computes the VSWF expansion coefficients for a monochromatic plane wave with
    the propagation direction given by the incident angles
    ``(polar_angle, azimuthal_angle)``.

    The coefficient mapping uses :func:`yasfpy.functions.misc.transformation_coefficients`
    and the associated scalar spherical functions from
    :func:`yasfpy.functions.spherical_functions_trigon.spherical_functions_trigon`.
    """

    lmax = sim.numerics.lmax
    E0 = sim.parameters.initial_field.amplitude
    k = sim.parameters.k_medium

    beta = sim.parameters.initial_field.polar_angle
    cb = np.cos(beta)
    sb = np.sin(beta)
    alpha = sim.parameters.initial_field.azimuthal_angle

    pilm, taulm = spherical_functions_trigon(lmax, beta)

    relative_particle_positions = (
        sim.parameters.particles.position - sim.parameters.initial_field.focal_point
    )
    kvec = np.outer(np.array((sb * np.cos(alpha), sb * np.sin(alpha), cb)), k)
    eikr = np.exp(1j * np.matmul(relative_particle_positions, kvec))

    sim.initial_field_coefficients = np.zeros(
        (
            sim.parameters.particles.number,
            sim.numerics.nmax,
            sim.parameters.k_medium.size,
        ),
        dtype=complex,
    )
    for m in range(-lmax, lmax + 1):
        for tau in range(1, 3):
            for l in range(np.max([1, np.abs(m)]), lmax + 1):
                n = multi2single_index(0, tau, l, m, lmax)
                sim.initial_field_coefficients[:, n, :] = (
                    4
                    * E0
                    * np.exp(-1j * m * alpha)
                    * eikr
                    * transformation_coefficients(
                        pilm,
                        taulm,
                        tau,
                        l,
                        m,
                        sim.parameters.initial_field.pol,
                        dagger=True,
                    )
                )


def _compute_wavebundle_normal_incidence(sim: "Simulation") -> None:
    r"""Compute a normal-incidence Gaussian wavebundle coefficient set.

    Parameters
    ----------
    sim:
        Simulation instance. The result is stored on
        ``sim.initial_field_coefficients``.

    Notes
    -----
    This implementation assumes a single wavelength (it uses
    ``k = sim.parameters.k_medium[0]``).

    The algorithm approximates a Gaussian beam via a weighted superposition of
    plane waves over polar angles (a "wavebundle" construction). It is primarily
    included for compatibility with CELES-style workflows :cite:`Egel-2017-ID1`.
    """

    from scipy.special import jv as besselj

    lmax = sim.numerics.lmax
    E0 = sim.parameters.initial_field.amplitude
    k = sim.parameters.k_medium[0]  # Assuming single wavelength for now
    w = sim.parameters.initial_field.beam_width
    prefac = E0 * k**2 * w**2 / np.pi

    pol_lower = sim.parameters.initial_field.polarization.lower()
    if pol_lower == "te":
        alphaG = sim.parameters.initial_field.azimuthal_angle
    elif pol_lower == "tm":
        alphaG = sim.parameters.initial_field.azimuthal_angle - np.pi / 2
    else:
        raise ValueError(
            f"Unsupported polarization: {sim.parameters.initial_field.polarization}"
        )

    full_beta_array = sim.numerics.polar_angles
    if full_beta_array is None:
        raise ValueError(
            "numerics.polar_angles not initialized. Set sampling_points_number in Numerics."
        )

    incident_polar = sim.parameters.initial_field.polar_angle
    direction_idcs = np.sign(np.cos(full_beta_array)) == np.sign(np.cos(incident_polar))
    beta_array = full_beta_array[direction_idcs]

    if len(beta_array) < 2:
        raise ValueError("Need at least 2 polar angles for wavebundle integration")
    dBeta = np.mean(np.diff(np.sort(beta_array)))

    cb = np.cos(beta_array)
    sb = np.sin(beta_array)

    gaussfac = np.exp(-(w**2) / 4 * k**2 * sb**2)
    gaussfacSincos = gaussfac * cb * sb

    pilm, taulm = spherical_functions_trigon(lmax, cb, sb)

    relative_particle_positions = (
        sim.parameters.particles.position - sim.parameters.initial_field.focal_point
    )

    rhoGi = np.sqrt(
        relative_particle_positions[:, 0] ** 2 + relative_particle_positions[:, 1] ** 2
    )
    phiGi = np.arctan2(
        relative_particle_positions[:, 1], relative_particle_positions[:, 0]
    )
    zGi = relative_particle_positions[:, 2]

    sim.initial_field_coefficients = np.zeros(
        (
            sim.parameters.particles.number,
            sim.numerics.nmax,
            sim.parameters.k_medium.size,
        ),
        dtype=complex,
    )

    for m in range(-lmax, lmax + 1):
        phase_m_minus_1 = np.exp(-1j * (m - 1) * phiGi)
        phase_m_plus_1 = np.exp(-1j * (m + 1) * phiGi)

        exp_ikz = np.exp(1j * zGi[:, np.newaxis] * k * cb[np.newaxis, :])

        bessel_m_minus_1 = besselj(
            np.abs(m - 1), rhoGi[:, np.newaxis] * k * sb[np.newaxis, :]
        )
        bessel_m_plus_1 = besselj(
            np.abs(m + 1), rhoGi[:, np.newaxis] * k * sb[np.newaxis, :]
        )

        radial_m_minus_1 = phase_m_minus_1[:, np.newaxis] * exp_ikz * bessel_m_minus_1
        radial_m_plus_1 = phase_m_plus_1[:, np.newaxis] * exp_ikz * bessel_m_plus_1

        eikzI1 = np.pi * (
            np.exp(-1j * alphaG) * (1j ** np.abs(m - 1)) * radial_m_minus_1
            + np.exp(1j * alphaG) * (1j ** np.abs(m + 1)) * radial_m_plus_1
        )

        eikzI2 = (
            np.pi
            * 1j
            * (
                -np.exp(-1j * alphaG) * (1j ** np.abs(m - 1)) * radial_m_minus_1
                + np.exp(1j * alphaG) * (1j ** np.abs(m + 1)) * radial_m_plus_1
            )
        )

        for tau in range(1, 3):
            for l in range(max(1, abs(m)), lmax + 1):
                n = multi2single_index(0, tau, l, m, lmax)

                gaussSincosBDag1 = gaussfacSincos * transformation_coefficients(
                    pilm, taulm, tau, l, m, 1, dagger=True
                )
                gaussSincosBDag2 = gaussfacSincos * transformation_coefficients(
                    pilm, taulm, tau, l, m, 2, dagger=True
                )

                integral = (
                    (
                        np.dot(eikzI1[:, 1:], gaussSincosBDag1[1:])
                        + np.dot(eikzI1[:, :-1], gaussSincosBDag1[:-1])
                        + np.dot(eikzI2[:, 1:], gaussSincosBDag2[1:])
                        + np.dot(eikzI2[:, :-1], gaussSincosBDag2[:-1])
                    )
                    * dBeta
                    / 2
                )

                sim.initial_field_coefficients[:, n, 0] = prefac * integral
