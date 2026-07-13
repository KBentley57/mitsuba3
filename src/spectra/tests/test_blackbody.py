import pytest
import drjit as dr
import mitsuba as mi


def interaction(wavelengths):
    si = mi.SurfaceInteraction3f()
    si.wavelengths = wavelengths
    return si


def blackbody(**kwargs):
    return mi.load_dict({
        "type": "blackbody",
        "temperature": 5000,
        **kwargs,
    })


def curve(values=None):
    if values is None:
        values = [0.2, 0.4, 0.8, 0.6]
    return {
        "type": "spectrum",
        "value": list(zip([360, 500, 700, 830], values)),
    }


def test01_default_and_scalar_fast_path(variant_scalar_spectral):
    si = interaction([360, 500, 700, 830])
    base = blackbody()
    grey = blackbody(emissivity=0.25)

    assert dr.allclose(base.eval(si), [6655.12, 12107.2, 11812.0, 9741.79])
    assert dr.allclose(grey.eval(si), 0.25 * base.eval(si))
    assert dr.allclose(grey.mean(), 0.25 * base.mean())
    assert dr.allclose(grey.max(), 0.25 * base.max())


def test02_scalar_clamping_and_traversal(variant_scalar_spectral):
    si = interaction([400, 500, 600, 700])
    base = blackbody()

    assert dr.allclose(blackbody(emissivity=-0.5).eval(si), 0)
    assert dr.allclose(blackbody(emissivity=2.0).eval(si), base.eval(si))

    grey = blackbody(emissivity=0.2)
    params = mi.traverse(grey)
    assert "emissivity" in params

    params["emissivity"] = 0.6
    params.update()
    assert dr.allclose(grey.eval(si), 0.6 * base.eval(si))

    params["emissivity"] = 4.0
    params.update()
    assert dr.allclose(grey.eval(si), base.eval(si))


def test03_inline_spectral_emissivity(variant_scalar_spectral):
    si = interaction([360, 500, 700, 830])
    base = blackbody()
    spectral = blackbody(emissivity=curve())

    assert dr.allclose(
        spectral.eval(si),
        base.eval(si) * mi.UnpolarizedSpectrum([0.2, 0.4, 0.8, 0.6]),
    )

    clamped = blackbody(emissivity=curve([-0.2, 0.4, 1.2, 0.6]))
    assert dr.allclose(
        clamped.eval(si),
        base.eval(si) * mi.UnpolarizedSpectrum([0.0, 0.4, 1.0, 0.6]),
    )


def test04_filename_spectrum_and_resolver(variant_scalar_spectral, tmp_path):
    spectrum_file = tmp_path / "emissivity.spd"
    spectrum_file.write_text(
        "# wavelength[nm] emissivity\n"
        "360 0.2\n"
        "500 0.4\n"
        "700 0.8\n"
        "830 0.6\n",
        encoding="utf-8",
    )

    previous_resolver = mi.file_resolver()
    resolver = mi.FileResolver(previous_resolver)
    resolver.prepend(str(tmp_path))
    mi.set_file_resolver(resolver)
    try:
        from_file = blackbody(emissivity={
            "type": "spectrum",
            "filename": spectrum_file.name,
        })
    finally:
        mi.set_file_resolver(previous_resolver)

    inline = blackbody(emissivity=curve())
    si = interaction([360, 500, 700, 830])
    assert dr.allclose(from_file.eval(si), inline.eval(si))


def test05_pdf_and_sample_weight_are_unbiased(variant_scalar_spectral):
    base = blackbody()
    spectral = blackbody(emissivity=curve())
    si = interaction([400, 500, 600, 700])

    # Emissivity changes the integrand, not the wavelength proposal density.
    assert dr.allclose(spectral.pdf_spectrum(si), base.pdf_spectrum(si))

    for sample in [0.05, 0.25, 0.5, 0.75, 0.95]:
        wav_base, _ = base.sample_spectrum(si, sample)
        wav, weight = spectral.sample_spectrum(si, sample)
        assert dr.allclose(wav, wav_base)

        si_sampled = interaction(wav)
        assert dr.allclose(
            weight * spectral.pdf_spectrum(si_sampled),
            spectral.eval(si_sampled),
            rtol=2e-5,
        )


def test06_estimator_matches_independent_quadrature(variant_scalar_spectral):
    import numpy as np

    spectral = blackbody(emissivity=curve())
    wavelengths_nm = np.linspace(360.0, 830.0, 20001)
    wavelengths_m = wavelengths_nm * 1e-9
    emissivity = np.interp(
        wavelengths_nm,
        [360.0, 500.0, 700.0, 830.0],
        [0.2, 0.4, 0.8, 0.6],
    )

    c = 2.99792458e8
    h = 6.62607004e-34
    k = 1.38064852e-23
    planck = 1e-9 * (2.0 * h * c * c) / (
        wavelengths_m**5 * (np.exp(h * c / (wavelengths_m * k * 5000.0)) - 1.0)
    )
    reference_integral = np.trapezoid(planck * emissivity, wavelengths_nm)

    assert float(spectral.mean()) == pytest.approx(
        reference_integral / (830.0 - 360.0), rel=3e-4
    )

    si = interaction([400, 500, 600, 700])
    sample_count = 512
    weights = []
    for i in range(sample_count):
        _, weight = spectral.sample_spectrum(
            si, (i + 0.5) / sample_count
        )
        weights.append(float(dr.mean(weight)))

    assert np.mean(weights) == pytest.approx(reference_integral, rel=2e-3)


def test07_temperature_texture_cross_product(variant_scalar_spectral):
    si = interaction([400, 500, 600, 700])
    scalar_temperature = blackbody(emissivity=0.35)
    texture_temperature = mi.load_dict({
        "type": "blackbody",
        "temperature": {"type": "uniform", "value": 5000},
        "emissivity": 0.35,
    })

    assert dr.allclose(texture_temperature.eval(si), scalar_temperature.eval(si))


def test08_validation(variant_scalar_spectral, tmp_path):
    with pytest.raises(RuntimeError, match="Use a float for constant emissivity"):
        blackbody(emissivity={"type": "spectrum", "value": 0.5})

    with pytest.raises(RuntimeError, match="does not cover"):
        blackbody(emissivity={
            "type": "spectrum",
            "value": [(400, 0.2), (800, 0.7)],
        })

    with pytest.raises(RuntimeError, match="must be finite"):
        blackbody(emissivity={
            "type": "spectrum",
            "value": [(360, 0.2), (500, float("nan")), (830, 0.7)],
        })

    with pytest.raises(RuntimeError, match="must be homogeneous"):
        blackbody(emissivity={
            "type": "checkerboard",
            "color0": 0.2,
            "color1": 0.8,
        })

    malformed = tmp_path / "malformed.spd"
    malformed.write_text("500 0.4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="at least two"):
        blackbody(emissivity={
            "type": "spectrum",
            "filename": str(malformed),
        })


def test09_spectral_traversal(variant_scalar_spectral):
    si = interaction([360, 500, 700, 830])
    spectral = blackbody(emissivity=curve())
    params = mi.traverse(spectral)

    assert "emissivity.values" in params
    params["emissivity.values"] = [0.1, 0.2, 0.3, 0.4]
    params.update()

    assert dr.allclose(
        spectral.eval(si),
        blackbody().eval(si) * mi.UnpolarizedSpectrum([0.1, 0.2, 0.3, 0.4]),
    )


def test10_area_emitter_integration(variant_scalar_spectral):
    shape = mi.load_dict({
        "type": "rectangle",
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "blackbody",
                "temperature": 5000,
                "emissivity": 0.3,
            },
        },
    })

    si = interaction([400, 500, 600, 700])
    si.wi = [0, 0, 1]
    assert dr.allclose(shape.emitter().eval(si), 0.3 * blackbody().eval(si))


def test11_vector_variant(variants_vec_spectral):
    si = interaction([400, 500, 600, 700])
    assert dr.allclose(
        blackbody(emissivity=0.4).eval(si),
        0.4 * blackbody().eval(si),
    )
