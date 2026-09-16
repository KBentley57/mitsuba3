import numpy as np
import drjit as dr
import mitsuba as mi


def test_camera_motion_over_static_thermal_emitter(variants_all_spectral):
    """Camera shutter integration must work with a stationary AVMC emitter."""
    T = mi.ScalarTransform4f

    def camera_transform(x):
        return T.look_at(origin=[x, 0, -3], target=[x, 0, 0], up=[0, 1, 0])

    def scene(animated):
        return mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 1},
            'sensor': {
                'type': 'perspective', 'fov': 1,
                'to_world': mi.AnimatedTransform4f({
                    0.0: camera_transform(-2), 1.0: camera_transform(2)
                }) if animated else camera_transform(0),
                'shutter_open': 0, 'shutter_close': 1 if animated else 0,
                'sampler': {'type': 'independent', 'sample_count': 4096},
                'film': {'type': 'specfilm', 'width': 1, 'height': 1,
                         'rfilter': {'type': 'box'},
                         'srf_lwir': {'type': 'regular', 'wavelength_min': 8000,
                                      'wavelength_max': 14000, 'values': '1,1'}},
            },
            'target': {
                'type': 'rectangle', 'to_world': T.scale([0.5, 0.5, 1]),
                'emitter': {'type': 'twosidedarea', 'radiance': {
                    'type': 'blackbody', 'temperature': 300,
                    'wavelength_min': 8000, 'wavelength_max': 14000}},
            },
        })

    moving, frozen = scene(True), scene(False)
    sensor = moving.sensors()[0]
    for time, hit in [(0, False), (0.5, True), (1, False)]:
        ray, _ = sensor.sample_ray(time, 0.3, [0.5, 0.5], [0.5, 0.5])
        assert dr.allclose(ray.time, time)
        si = moving.ray_intersect(ray)
        assert bool(dr.all(si.is_valid())) == hit
        if hit:
            assert dr.all(si.emitter(moving).eval(si) > 0)

    blurred = float(np.array(mi.render(moving, seed=17, spp=4096)).item())
    static = float(np.array(mi.render(frozen, seed=17, spp=4096)).item())
    assert static > 0
    # A width-1 stationary target occupies one quarter of a width-4 sweep.
    assert abs(blurred / static - 0.25) < 0.025
