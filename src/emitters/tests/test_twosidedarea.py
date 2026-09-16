import drjit as dr
import mitsuba as mi
import pytest
import numpy as np


@pytest.mark.parametrize("textured", [False, True])
def test_matches_upstream_twosided_area(variants_all_rgb, textured):
    """Keep the AVMC plugin name and both hemispheres of its emission API."""
    def shape(plugin):
        emitter = {"type": plugin, "radiance": 3.0}
        if textured:
            emitter['radiance'] = {'type': 'bitmap',
                                   'bitmap': mi.Bitmap(np.full((4, 4, 3), 3, dtype=np.float32))}
        if plugin == "area":
            emitter["twosided"] = True
        return mi.load_dict({"type": "rectangle", "emitter": emitter,
                             "to_world": mi.ScalarTransform4f().translate([0, 0, 1000])})

    custom_shape, reference_shape = shape("twosidedarea"), shape("area")
    custom, reference = custom_shape.emitter(), reference_shape.emitter()
    for side in (-1, 1):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = [0, 0, side]
        si.p = [0, 0, 1000 + 2 * side]
        assert dr.allclose(custom.eval(si), reference.eval(si))
        ds, weight = custom.sample_direction(si, [0.3, 0.7])
        ds_ref, weight_ref = reference.sample_direction(si, [0.3, 0.7])
        assert dr.allclose(ds.p_err, ds_ref.p_err)
        assert dr.all(ds.p_err > 0)
        assert dr.allclose(ds.pdf, ds_ref.pdf)
        assert dr.allclose(weight, weight_ref)
        assert dr.allclose(custom.pdf_direction(si, ds), ds.pdf)

    for x in (0.2, 0.7):
        ray, weight = custom.sample_ray(0.3, 0.4, [0.3, 0.7], [x, 0.6])
        ray_ref, weight_ref = reference.sample_ray(0.3, 0.4, [0.3, 0.7], [x, 0.6])
        assert dr.allclose(ray.o, ray_ref.o)
        assert dr.allclose(ray.d, ray_ref.d)
        assert dr.allclose(weight, weight_ref)
        assert dr.all(weight > 0)
        assert dr.all(ray.d.z * (1 if x < 0.5 else -1) > 0)
