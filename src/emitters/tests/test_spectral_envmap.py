import math
import numpy as np
import pytest
import drjit as dr
import mitsuba as mi


def emitter(data=None, **kwargs):
    if data is None:
        data = np.ones((3, 4, 3), dtype=np.float32)
    return mi.load_dict({"type": "spectral_envmap", "bitmap": mi.Bitmap(data),
                         "wavelengths": "8000, 10000, 12000", **kwargs})


def interaction(wavelengths=(8000, 9000, 10000, 12000), direction=(0, 0, -1)):
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = wavelengths
    si.wi = -mi.Vector3f(direction)
    return si


def test_interpolation_and_support(variants_all_spectral):
    data = np.broadcast_to([1, 3, 7], (3, 4, 3)).astype(np.float32).copy()
    e = emitter(data)
    assert dr.allclose(e.eval(interaction()), [1, 2, 3, 7])
    assert dr.allclose(e.eval(interaction([7999, 8000, 12000, 12001])), [0, 1, 7, 0])


def test_named_exr_channels(variant_scalar_spectral, tmp_path):
    names = ['band_a', 'band_b', 'band_c', 'band_d', 'band_e']
    data = np.broadcast_to([1, 2, 4, 8, 16], (3, 4, 5)).astype(np.float16).copy()
    bitmap = mi.Bitmap(data, channel_names=names)
    path = tmp_path / 'named.exr'
    bitmap.write(str(path))
    e = mi.load_dict({'type': 'spectral_envmap', 'filename': str(path),
                     'wavelengths': '8000, 9000, 10000, 11000, 12000'})
    assert dr.allclose(e.eval(interaction([8000, 9000, 11000, 12000])), [1, 2, 8, 16])


@pytest.mark.parametrize('data', [np.full((3, 4, 3), -1, dtype=np.float32),
                                  np.full((3, 4, 3), np.inf, dtype=np.float32),
                                  np.full((3, 4, 3), np.nan, dtype=np.float32),
                                  np.ones((2, 4, 3), dtype=np.float32)])
def test_invalid_map(variant_scalar_spectral, data):
    with pytest.raises(RuntimeError):
        emitter(data)


@pytest.mark.parametrize('kwargs', [
    {'wavelengths': '8000,10000foo,12000'},
    {'wavelengths': '8000,nan,12000'},
    {'scale': float('inf')},
    {'sun_radiance': 1, 'sun_direction': [0, 0, 0]},
    {'sun_radiance': 1, 'sun_direction': [0, 1, 0], 'sun_half_aperture': float('nan')},
    {'sun_radiance': 1, 'sun_direction': [0, 1, 0], 'sun_scale': -1},
])
def test_invalid_parameters(variant_scalar_spectral, kwargs):
    with pytest.raises(RuntimeError):
        emitter(**kwargs)


def test_update_halo_and_resize(variants_all_spectral):
    e = emitter()
    params = mi.traverse(e)
    data = np.zeros((3, 5, 8, 1), dtype=np.float32)
    data[:, :, 1:-1, :] = 2
    data[:, :, 1, :] = 4
    data[:, :, -2, :] = 8
    data *= np.array([1, 2, 4], dtype=np.float32)[:, None, None, None]
    params['data'] = mi.TensorXf(data)
    params.update()
    actual = np.array(params['data'])
    assert np.allclose(actual[:, :, 0], np.array([8, 16, 32])[:, None, None])
    assert np.allclose(actual[:, :, -1], np.array([4, 8, 16])[:, None, None])
    # At phi=0 the two periodic edge texels are interpolated equally.
    assert dr.allclose(e.eval(interaction()), [6, 9, 12, 24])
    assert dr.allclose(e.eval(interaction(direction=(0, 0, 1))), [2, 3, 4, 8])
    ds, w = e.sample_direction(interaction(), [0.37, 0.63])
    assert dr.allclose(ds.pdf, e.pdf_direction(interaction(), ds), rtol=1e-4)
    assert dr.allclose(w * ds.pdf, e.eval_direction(interaction(), ds), rtol=1e-4)
    fresh = emitter(np.moveaxis(data[:, :, 1:-1, 0], 0, -1).copy())
    ds_fresh, _ = fresh.sample_direction(interaction(), [0.37, 0.63])
    assert dr.allclose(ds.d, ds_fresh.d)
    assert dr.allclose(ds.pdf, ds_fresh.pdf)
    wl, sw = e.sample_wavelengths(interaction(), 0.31)
    wl_fresh, sw_fresh = fresh.sample_wavelengths(interaction(), 0.31)
    assert dr.allclose(wl, wl_fresh)
    assert dr.allclose(sw, sw_fresh)
    params['data'] = mi.TensorXf(np.ones((4, 5, 8, 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match='wavelength'):
        params.update()


def test_halo_gradient_routes_to_real_texels(variant_llvm_ad_spectral):
    e = emitter()
    params = mi.traverse(e)
    leaf = mi.TensorXf(np.ones((3, 3, 6, 1), dtype=np.float32))
    dr.enable_grad(leaf)
    params['data'] = leaf
    params.update()
    value = e.eval(interaction([8000] * 4))[0]
    dr.backward(value)
    grad = np.array(dr.grad(leaf))
    assert np.allclose(grad[0, 1, 1, 0], 0.5)
    assert np.allclose(grad[0, 1, 4, 0], 0.5)
    assert np.all(grad[:, :, (0, -1), :] == 0)
    assert np.isclose(grad.sum(), 1)


def sun_spectrum():
    return {'type': 'regular', 'wavelength_min': 14000, 'wavelength_max': 16000,
            'values': '2,2'}


def test_sun_only_wavelength_support(variant_llvm_ad_spectral):
    e = emitter(np.zeros((3, 4, 3), dtype=np.float32),
                sun_radiance=sun_spectrum(), sun_direction=[0, 0, -1])
    si = interaction([13000, 14000, 15000, 16000])
    assert dr.allclose(e.eval(si), [0, 2, 2, 2])
    si.uv = [0, 0.5]
    u = (dr.arange(mi.Float, 16384) + 0.5) / 16384
    wavelengths, weights = e.sample_wavelengths(si, u)
    assert dr.all(dr.isfinite(weights), axis=None)
    assert dr.any(wavelengths[0] > 14000)
    # Integral of the supplied sun radiance is 2 * 2000 nm.
    assert dr.allclose(dr.mean(weights[0]), 4000, rtol=2e-3)


@pytest.mark.parametrize('radius', [0.2665, 30])
def test_direct_normal_irradiance(variants_all_spectral, radius):
    irradiance = 2 * math.pi * math.sin(math.radians(radius)) ** 2
    e = emitter(np.zeros((3, 4, 3), dtype=np.float32),
                sun_irradiance={'type': 'uniform', 'value': irradiance,
                                'wavelength_min': 8000, 'wavelength_max': 12000},
                sun_direction=[0, 0, -1], sun_half_aperture=radius)
    assert dr.allclose(e.eval(interaction()), 2, rtol=1e-5)
    assert dr.allclose(e.eval(interaction(direction=(0, 0, 1))), 0)


def test_dark_map_and_rotated_emission(variants_all_spectral):
    for level in (0, 1):
        e = emitter(np.full((3, 4, 3), level, dtype=np.float32),
                    to_world=mi.ScalarTransform4f().rotate([1, 0, 0], 63))
        ray, weight = e.sample_ray(0.4, 0.3, [0.2, 0.7], [0.31, 0.67])
        assert dr.all(dr.isfinite(weight))
        assert dr.allclose(dr.dot(ray.o, ray.d), -1, atol=1e-5)
        if level == 0:
            assert dr.allclose(weight, 0)


@pytest.mark.parametrize('radius', [0.2665, 30])
def test_sun_direction_sampling_integrates_irradiance(variant_llvm_ad_spectral, radius):
    e = emitter(np.zeros((3, 4, 3), dtype=np.float32),
                sun_radiance={'type': 'uniform', 'value': 2,
                              'wavelength_min': 8000, 'wavelength_max': 12000},
                sun_direction=[0, 0, -1], sun_half_aperture=radius)
    n = 32768
    rng = mi.PCG32(size=n)
    sample = mi.Point2f((dr.arange(mi.Float, n) + 0.5) / n, rng.next_float32())
    si = interaction([10000] * 4)
    ds, weight = e.sample_direction(si, sample)
    assert dr.all(dr.isfinite(weight), axis=None)
    assert dr.allclose(ds.pdf, e.pdf_direction(si, ds), rtol=1e-4)
    irradiance = dr.mean(weight[0] * dr.maximum(-ds.d.z, 0))
    expected = 2 * math.pi * math.sin(math.radians(radius)) ** 2
    assert dr.allclose(irradiance, expected, rtol=0.02)


def test_animated_environment_rotation(variants_all_spectral):
    e = emitter(np.zeros((3, 4, 3), dtype=np.float32),
                sun_radiance={'type': 'uniform', 'value': 2,
                              'wavelength_min': 8000, 'wavelength_max': 12000},
                sun_direction=[0, 0, -1],
                to_world=mi.AnimatedTransform4f({
                    0.0: mi.ScalarTransform4f(),
                    1.0: mi.ScalarTransform4f().rotate([0, 1, 0], 90)}))
    si = interaction()
    assert dr.allclose(e.eval(si), 2)
    si.time = 1
    assert dr.allclose(e.eval(si), 0)
    si.wi = [1, 0, 0]
    assert dr.allclose(e.eval(si), 2)
    ds, weight = e.sample_direction(si, [0.1, 0.6])
    assert dr.allclose(ds.pdf, e.pdf_direction(si, ds), rtol=1e-4)
    assert dr.all(weight > 0)
    assert dr.all(ds.d.x < -0.99)


def test_mis_compensation_keeps_emission_support(variant_llvm_ad_spectral):
    data = np.ones((5, 8, 3), dtype=np.float32)
    data[:, 3:5, :] = 20
    a = emitter(data, mis_compensation=False)
    b = emitter(data, mis_compensation=True)
    rng = mi.PCG32(size=4096)
    sample = mi.Point2f(rng.next_float32(), rng.next_float32())
    ra, wa = a.sample_ray(0.0, 0.3, [0.2, 0.7], sample)
    rb, wb = b.sample_ray(0.0, 0.3, [0.2, 0.7], sample)
    assert dr.allclose(ra.d, rb.d, atol=2e-5)
    assert dr.allclose(wa, wb, rtol=1e-4)


@pytest.mark.parametrize('invalid', [-1, float('nan'), float('inf')])
def test_invalid_map_update(variants_all_spectral, invalid):
    e = emitter()
    params = mi.traverse(e)
    data = np.ones((3, 3, 6, 1), dtype=np.float32)
    data[1, 1, 2, 0] = invalid
    params['data'] = mi.TensorXf(data)
    with pytest.raises(RuntimeError, match='finite, non-negative'):
        params.update()


def test_lwir_render_sees_sky_and_sun(variants_all_spectral):
    e = emitter(np.full((3, 4, 3), 2, dtype=np.float32),
                sun_radiance={'type': 'uniform', 'value': 3,
                              'wavelength_min': 8000, 'wavelength_max': 12000},
                sun_direction=[0, 0, -1])
    for direction, expected in [([0, 0, -1], 20000), ([0, 0, 1], 8000)]:
        scene = mi.load_dict({
            'type': 'scene', 'environment': e,
            'integrator': {'type': 'path'},
            'sensor': {
                'type': 'perspective', 'fov': 0.1,
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 0], target=direction, up=[0, 1, 0]),
                'film': {'type': 'specfilm', 'width': 1, 'height': 1,
                         'rfilter': {'type': 'box'},
                         'srf_lwir': {'type': 'regular', 'wavelength_min': 8000,
                                      'wavelength_max': 12000, 'values': '1,1'}},
            },
        })
        value = float(np.array(mi.render(scene, seed=7, spp=16)).item())
        assert np.isclose(value, expected, rtol=1e-5)


def test_scale_update_preserves_gradient(variant_llvm_ad_spectral):
    e = emitter()
    params = mi.traverse(e)
    scale = mi.Float(2)
    dr.enable_grad(scale)
    params['scale'] = scale
    params.update()
    value = e.eval(interaction())[0]
    assert dr.allclose(value, 2)
    dr.backward(value)
    assert dr.allclose(dr.grad(scale), 1)
    params['scale'] = 0
    params.update()
    assert dr.allclose(e.eval(interaction()), 0)
