.. _thermal-emission:

Thermal emission and emissivity
===============================

This page describes the radiometric model and wavelength-sampling estimator
used by Mitsuba's :ref:`spectrum-blackbody` spectrum. The model is intended for
surfaces in local thermal equilibrium whose temperature and emissivity are
known independently of the BSDF.

Planck spectral radiance
------------------------

The spectral radiance of an ideal blackbody at absolute temperature :math:`T`
is given by Planck's law,

.. math::

    B_\lambda(T) =
    \frac{2 h c^2}{\lambda^5}
    \frac{1}{\exp\!\left(\frac{h c}{\lambda k T}\right)-1},

where :math:`h` is the Planck constant, :math:`c` is the speed of light, and
:math:`k` is the Boltzmann constant. Mitsuba accepts wavelengths in nanometers
and applies the required :math:`10^{-9}` conversion so that the returned units
are :math:`\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}}`.

Real surfaces
-------------

A real surface emits a fraction of the ideal blackbody radiance. Its emitted
spectral radiance is

.. math::

    L_\lambda(T) = \epsilon_\lambda B_\lambda(T),
    \qquad 0 \leq \epsilon_\lambda \leq 1.

Mitsuba supports either a scalar greybody emissivity :math:`\epsilon` or one
homogeneous spectral curve :math:`\epsilon_\lambda` per blackbody spectrum.
A scalar explicitly selects the constant fast path. Spectral data can be
provided inline or read from a standard ``.spd`` file, with one wavelength in
nanometers and one emissivity value per line. The curve must cover the complete
configured rendering wavelength interval. Values outside the physical interval
are clamped to :math:`[0,1]`.

The initial implementation deliberately excludes spatial and per-vertex
emissivity. A tabulated curve is always evaluated spectrally, even if all of
its samples happen to be equal; no heuristic flatness test is performed.

Finite wavelength interval
--------------------------

For wavelength bounds :math:`[\lambda_{\min},\lambda_{\max}]`, a sensor or
integrator estimates quantities involving

.. math::

    I(T) = \int_{\lambda_{\min}}^{\lambda_{\max}}
           \epsilon_\lambda B_\lambda(T)\,\mathrm{d}\lambda.

The emissivity curve must cover this entire interval. Otherwise a portion of
the integrand would be undefined, and silently treating it as zero would change
the modeled material and bias the result.

Wavelength sampling
-------------------

The blackbody plugin samples a wavelength from its existing temperature-based
proposal density :math:`p_B(\lambda\mid T)`. Emissivity does not change that
proposal. For a sampled wavelength :math:`\lambda_i`, the returned Monte Carlo
weight is

.. math::

    w_i = \frac{\epsilon_{\lambda_i} B_{\lambda_i}(T)}
                {p_B(\lambda_i\mid T)}.

This remains unbiased because

.. math::

    \mathbb{E}[w_i]
    = \int p_B(\lambda\mid T)
      \frac{\epsilon_\lambda B_\lambda(T)}{p_B(\lambda\mid T)}
      \,\mathrm{d}\lambda
    = I(T),

provided that :math:`p_B` is nonzero wherever
:math:`\epsilon_\lambda B_\lambda(T)` is nonzero. The proposal is based on
Wien's approximation while evaluation uses Planck's law. This mismatch can
change variance, but not the expectation or correctness of the estimator.
Sampling directly from :math:`\epsilon_\lambda B_\lambda(T)` could reduce
variance for sharply structured emissivity curves, but would require rebuilding
or selecting a material-specific distribution and is not part of this design.

Energy conservation and the BSDF
--------------------------------

For passive material response at a wavelength, the incident-energy fractions
satisfy

.. math::

    \rho_\lambda + \tau_\lambda + \alpha_\lambda = 1,

where :math:`\rho`, :math:`\tau`, and :math:`\alpha` denote reflectivity,
transmissivity, and absorptivity. Under local thermal equilibrium, Kirchhoff's
law gives

.. math::

    \epsilon_\lambda = \alpha_\lambda.

These relationships belong to the material definition and its validation.
Mitsuba's blackbody spectrum only applies emissivity to emitted radiance; it
does not alter the BSDF or automatically derive reflection, transmission, or
absorption. A physically consistent SGX material should therefore provide both
the BSDF model and the wavelength-dependent radiometric parameters, and should
validate their conservation relationship before constructing the Mitsuba
objects.
