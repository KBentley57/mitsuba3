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

Spectral sky and sun inputs
--------------------------

The :ref:`emitter-spectral_envmap` emitter represents a distant sky with a
multi-channel equirectangular radiance map and an optional uniform sun disk.
It reads supplied spectra; it does not compute an atmosphere or automatically
invoke the blackbody plugin. A blackbody spectrum can be supplied explicitly
as the sun input if that is the intended model.

Map values are linear spectral radiance in
:math:`\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}}`, sampled on a uniform wavelength
grid in nanometers. The wavelength list maps positionally to the **loaded**
Bitmap channels. Inspect ``bitmap.struct_()`` when loading named EXR channels:
lexical channel ordering is not necessarily numeric wavelength ordering.
There must be at least two wavelength channels, two columns, and three rows.

Supply either ``sun_radiance`` in the same radiance units or
``sun_irradiance`` in :math:`\mathrm{W\,m^{-2}\,nm^{-1}}`. The latter is direct
irradiance on a plane normal to the sun direction. For angular radius
:math:`\alpha`, the conversion is

.. math::

    E_\lambda = \int_{\text{disk}} L_\lambda \cos\theta\,d\omega
               = L_\lambda\,\pi\sin^2\alpha.

This uses projected solid angle, rather than the unprojected disk solid angle
:math:`2\pi(1-\cos\alpha)`. The disk adds to the sky map, so remove any solar
disk already present in the input if it should not be counted twice.

Sky evaluation is zero outside the map's tabulated range. Sun evaluation is
zero outside the range declared by its spectrum. For a ``uniform`` spectrum,
set ``wavelength_min`` and ``wavelength_max`` explicitly when modeling infrared
radiance; its default sampling range is visible. Emitted-ray wavelength
sampling mixes the mean sky spectrum with a uniform proposal over the union
of the two ranges, so sun-only wavelengths retain sampling support. Sensor
wavelength sampling remains controlled by the sensor and film.

``mi.traverse(emitter)['data']`` exposes a tensor with shape
``(wavelength_count, height, width+2, 1)``. Edit columns ``1:-1`` and call
``params.update()``. The emitter regenerates both periodic edge columns,
preserves their gradient connection to the real columns, and rebuilds its
sampling distributions. Spatial resolution can change; changing the wavelength
grid requires a new emitter instance. A zero map is valid, including when the
sun is the sole source of illumination.

Camera motion
-------------

Use ``mi.AnimatedTransform4f`` for the camera's ``to_world`` and set a nonzero
``shutter_open`` / ``shutter_close`` interval to integrate camera motion. The
``spectral_envmap`` also evaluates its rotation at each ray's time, keeping the
sky map and sun disk together. See :ref:`sec-animation` for keyframe syntax.

The integration tests cover a moving camera observing stationary thermal
geometry through the retained ``twosidedarea`` emitter. Moving emissive targets
and atmospheric attenuation/path radiance between a target and sensor require
separate modeling; the sky map does not perform that transport calculation.
