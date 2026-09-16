#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <drjit/tensor.h>
#include <drjit/texture.h>
#include <algorithm>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-spectral_envmap:

Spectral environment map emitter (:monosp:`spectral_envmap`)
------------------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Path to a multi-channel OpenEXR image. Each channel holds the radiance for
     one wavelength, in loaded Bitmap channel order, matching ``wavelengths``.

 * - bitmap
   - :monosp:`Bitmap`
   - An in-memory bitmap with one channel per wavelength, as an alternative to
     :monosp:`filename`. Convenient from Python.

 * - wavelengths
   - |string|
   - Comma-separated wavelengths in nanometres, one per channel. Must be
     **uniformly spaced** and monotonically increasing.

 * - scale
   - |float|
   - Multiplicative factor applied to the stored radiance. (Default: 1)

 * - mis_compensation
   - |bool|
   - Apply MIS compensation to direct-light direction sampling. Emitted rays
     use an uncompensated distribution to retain full support.
     (Default: |false|)

 * - to_world
   - |transform|
   - Rotation applied to the environment, sun included. (Default: identity)

 * - sun_radiance / sun_irradiance
   - |spectrum|
   - Enables an analytic sun disk. Give **exactly one**:
     :monosp:`sun_radiance` is the radiance of the disk (W·m⁻²·sr⁻¹·nm⁻¹),
     :monosp:`sun_irradiance` is what a surface normal to the sun receives
     (W·m⁻²·nm⁻¹) and is converted internally via :math:`L = E / (\pi \sin^2 \alpha)`. The
     two spellings are kept distinct because confusing them is the classic
     error here. Omit both for a sky-only environment.

 * - sun_direction
   - |vector|
   - Direction pointing **from the scene toward the sun**. Mutually exclusive
     with :monosp:`sun_elevation`/:monosp:`sun_azimuth`.

 * - sun_elevation, sun_azimuth
   - |float|
   - Sun position in degrees, as an alternative to :monosp:`sun_direction`.
     Elevation is measured up from the horizon; azimuth follows the map's own
     convention (+Y up, measured from −Z toward +X).

 * - sun_half_aperture
   - |float|
   - Angular radius of the disk in degrees. (Default: 0.2665, the sun as seen
     from Earth)

 * - sun_scale
   - |float|
   - Multiplicative factor applied to the sun. (Default: 1)

 * - sun_sampling_weight
   - |float|
   - Fraction of direction samples drawn from the sun cone rather than from the
     map, in (0, 1). Affects variance only, never correctness. Raise it when
     the sun dominates (short-wave bands), lower it when the sky does
     (thermal). (Default: 0.25)

This emitter stores **linear spectral radiance** on an equirectangular grid, in
units of W·m⁻²·sr⁻¹·nm⁻¹, and interpolates it in both direction and wavelength.

The sun lives here rather than in a plugin of its own because a scene may
contain only one environment emitter, and only that emitter is consulted for
rays that escape the scene. A separate sun plugin could illuminate the scene but
could never be *seen* by the sensor. :ref:`sunsky <emitter-sunsky>` is a single
plugin for the same reason.

It exists because :ref:`envmap <emitter-envmap>` cannot represent spectra
outside the visible range: that plugin converts its input to linear sRGB, stores
three coefficients of a spectral *upsampling* model plus a scale, multiplies the
reconstructed spectrum by the D65 illuminant, draws wavelengths from D65, and
builds its direction-sampling distribution from photometric luminance. Every one
of those steps is specific to human colour vision. This plugin stores measured
radiance directly and does none of them, which makes it suitable for infrared
work where the data come from an atmospheric radiative-transfer code.

Because the stored quantity is linear, evaluation is a single trilinear texture
fetch: interpolation over direction and wavelength commute, unlike the nonlinear
upsampling model in :ref:`envmap <emitter-envmap>`, which has to fetch four
corners and evaluate the model separately at each one.

Sky wavelengths outside the tabulated range evaluate to zero. The sun is
limited to the wavelength range declared by its spectrum; set that range
explicitly for a uniform spectrum used outside the visible band. Wavelength
sampling covers the union of both ranges.

Channel names are preserved during component conversion, without interpreting
them as wavelengths. Supply ``wavelengths`` in the order of the **loaded**
``Bitmap.struct_()`` fields: the EXR reader can sort names, and lexical order need
not be numeric wavelength order. Inputs must be linear, with finite non-negative
values, at least two columns and three rows.

The exposed ``data`` tensor has shape ``(wavelength_count, H, W+2, 1)``.
Edit the real columns ``1:-1`` and call ``params.update()``. Both periodic halo
columns are regenerated with gradients routed to their source columns, and the
sampling distributions are rebuilt. Spatial resolution may change; the number
and grid of wavelengths are fixed for each emitter.

.. tabs::
    .. code-tab:: xml

        <emitter type="spectral_envmap">
            <string name="filename" value="sky_lwir.exr"/>
            <string name="wavelengths" value="8000, 9000, 10000, 11000, 12000, 13000, 14000"/>

            <!-- optional analytic sun disk -->
            <float name="sun_elevation" value="35"/>
            <float name="sun_azimuth" value="120"/>
            <spectrum name="sun_irradiance" filename="solar_direct.spd"/>
        </emitter>

*/

template <typename Float, typename Spectrum>
class SpectralEnvironmentMapEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using Warp = Hierarchical2D<Float, 0>;
    /* Wavelength is the *third* texture dimension rather than a channel: the
       channel count of a dr::Texture must be known at compile time, whereas the
       number of tabulated wavelengths is a property of the data. Making it a
       spatial axis keeps it runtime-sized and lets the hardware interpolate it. */
    using Tex = dr::Texture<Float, 3>;
    using Texel = dr::Array<Float, 1>;

    SpectralEnvironmentMapEmitter(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>) {
            Throw("SpectralEnvironmentMapEmitter: this plugin stores spectral "
                  "radiance and requires a spectral variant (e.g. "
                  "'scalar_spectral'). The active variant is not spectral.");
        } else {
            m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);

            // ---- wavelength grid ----------------------------------------
            std::vector<std::string> tokens = string::tokenize(
                std::string(props.get<std::string_view>("wavelengths")), " ,");
            if (tokens.size() < 2)
                Throw("SpectralEnvironmentMapEmitter: \"wavelengths\" must list "
                      "at least two wavelengths in nanometres.");

            std::vector<ScalarFloat> wavelengths;
            wavelengths.reserve(tokens.size());
            for (const std::string &t : tokens) {
                size_t end = 0;
                ScalarFloat value;
                try {
                    value = (ScalarFloat) std::stod(t, &end);
                } catch (const std::exception &) {
                    Throw("Invalid wavelength token: %s", t);
                }
                if (end != t.size() || !std::isfinite(value) || value <= 0.f)
                    Throw("Wavelengths must be finite, positive nanometres: %s", t);
                wavelengths.push_back(value);
            }

            m_wavelength_min = wavelengths.front();
            m_wavelength_max = wavelengths.back();
            m_n_wavelengths  = (uint32_t) wavelengths.size();
            m_wavelength_step =
                (m_wavelength_max - m_wavelength_min) / (m_n_wavelengths - 1);

            if (!std::isfinite(m_wavelength_step) || m_wavelength_step <= 0)
                Throw("SpectralEnvironmentMapEmitter: \"wavelengths\" must be "
                      "monotonically increasing.");

            /* Uniform spacing is required because the wavelength axis is
               interpolated as a texture coordinate. MODTRAN-style data are
               uniform in wavenumber, so resample before getting here. */
            for (uint32_t i = 1; i < m_n_wavelengths; ++i) {
                ScalarFloat step = wavelengths[i] - wavelengths[i - 1];
                if (dr::abs(step - m_wavelength_step) >
                    1e-3f * m_wavelength_step)
                    Throw("SpectralEnvironmentMapEmitter: \"wavelengths\" must "
                          "be uniformly spaced (expected a step of %f nm, but "
                          "entries %u and %u differ by %f nm). Resample the "
                          "data onto a uniform grid in wavelength.",
                          m_wavelength_step, i - 1, i, step);
            }

            // ---- pixel data ---------------------------------------------
            ref<Bitmap> bitmap;
            if (props.has_property("bitmap")) {
                if (props.has_property("filename"))
                    Throw("SpectralEnvironmentMapEmitter: cannot specify both "
                          "\"bitmap\" and \"filename\".");
                ref<Object> other = props.get<ref<Object>>("bitmap");
                Bitmap *b = dynamic_cast<Bitmap *>(other.get());
                if (!b)
                    Throw("SpectralEnvironmentMapEmitter: property \"bitmap\" "
                          "must be a Bitmap instance.");
                bitmap = b;
            } else {
                FileResolver *fs = file_resolver();
                fs::path file_path =
                    fs->resolve(props.get<std::string_view>("filename"));
                m_filename = file_path.filename().string();
                bitmap = new Bitmap(file_path);
            }

            if (bitmap->channel_count() != m_n_wavelengths)
                Throw("SpectralEnvironmentMapEmitter: the image has %zu "
                      "channels but %u wavelengths were given. There must be "
                      "exactly one channel per wavelength.",
                      bitmap->channel_count(), m_n_wavelengths);

            /* Normalise the component type to ScalarFloat but keep the pixel
               format as-is. Converting the *format* would apply colour
               semantics: a bitmap with 1-4 channels is labelled Y/YA/RGB/RGBA,
               so a 4-band spectral image looks like RGBA and a format
               conversion would colour-manage it (and can emit negative
               values). Channels are read positionally below, which is correct
               regardless of the label, since the channel count is already
               required to equal the number of wavelengths. */
            if (bitmap->srgb_gamma())
                Throw("spectral_envmap requires linear radiance, not sRGB-encoded data.");
            std::vector<std::string> channel_names;
            for (const auto &field : bitmap->struct_())
                channel_names.push_back(field.name);
            ref<Bitmap> converted = new Bitmap(bitmap->pixel_format(),
                struct_type_v<ScalarFloat>, bitmap->size(),
                bitmap->channel_count(), channel_names);
            converted->set_srgb_gamma(false);
            converted->set_premultiplied_alpha(bitmap->premultiplied_alpha());
            bitmap->convert(converted.get());
            bitmap = converted;

            m_res = ScalarVector2u((uint32_t) bitmap->width(),
                                   (uint32_t) bitmap->height());
            if (m_res.x() < 2 || m_res.y() < 3)
                Throw("SpectralEnvironmentMapEmitter: the image must be at "
                      "least 2x3 pixels (got %ux%u).", m_res.x(), m_res.y());

            /* Storage layout is (wavelength, theta, phi) with a single channel.
               A one-column halo on each side of the phi axis carries a copy of
               the opposite edge, so ``WrapMode::Clamp`` reproduces the periodic
               boundary in azimuth exactly -- the same trick `envmap` uses. */
            const uint32_t sw = m_res.x() + 2;
            const size_t n = (size_t) m_n_wavelengths * m_res.y() * sw;
            std::unique_ptr<ScalarFloat[]> storage(new ScalarFloat[n]);

            const ScalarFloat *src = (const ScalarFloat *) bitmap->data();
            for (uint32_t y = 0; y < m_res.y(); ++y) {
                for (uint32_t x = 0; x < m_res.x(); ++x) {
                    const ScalarFloat *px =
                        src + ((size_t) y * m_res.x() + x) * m_n_wavelengths;
                    for (uint32_t w = 0; w < m_n_wavelengths; ++w) {
                        if (!std::isfinite(px[w]) || px[w] < 0.f)
                            Throw("SpectralEnvironmentMapEmitter: the image "
                                  "contains a negative or non-finite radiance "
                                  "at pixel (%u, %u), channel %u. Radiance "
                                  "must be non-negative and finite.", x, y, w);
                        // Real column x is stored at column x + 1.
                        storage[flat(w, y, x + 1)] = px[w];
                    }
                }
            }
            refresh_halo(storage.get(), m_res, m_n_wavelengths);

            TensorXf tensor(storage.get(), { (size_t) m_n_wavelengths,
                                             (size_t) m_res.y(), (size_t) sw, 1 });
            m_texture = Tex(tensor, /* use_accel = */ true,
                            dr::FilterMode::Linear, dr::WrapMode::Clamp);

            m_scale = dr::opaque<Float>(checked_scale(props, "scale"));
            m_sampling_min = m_wavelength_min;
            m_sampling_max = m_wavelength_max;
            m_mis_compensation = props.get<bool>("mis_compensation", false);
            m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;

            // ---- optional analytic sun disk -----------------------------
            /* The sun lives inside this plugin rather than in a separate one
               because a scene may only contain a single environment emitter
               (see Scene::Scene, "Only one environment emitter ..."), and only
               the environment emitter is evaluated for rays that escape the
               scene. A separate sun plugin could therefore never be *seen* by
               camera rays alongside the sky. `sunsky` is one plugin for the
               same reason. */
            bool has_radiance   = props.has_property("sun_radiance");
            bool has_irradiance = props.has_property("sun_irradiance");

            if (has_radiance && has_irradiance)
                Throw("SpectralEnvironmentMapEmitter: specify at most one of "
                      "\"sun_radiance\" (W/m^2/sr/nm, the radiance of the "
                      "disk) and \"sun_irradiance\" (W/m^2/nm, received by a "
                      "surface normal to the sun).");

            m_has_sun = has_radiance || has_irradiance;

            if (m_has_sun) {
                ScalarFloat half_aperture =
                    props.get<ScalarFloat>("sun_half_aperture", 0.2665f);
                if (!std::isfinite(half_aperture) || !(half_aperture > 0.f) || half_aperture >= 90.f)
                    Throw("SpectralEnvironmentMapEmitter: "
                          "\"sun_half_aperture\" must lie in (0, 90) degrees "
                          "(got %f).", half_aperture);

                /* Store 1 - cos(theta) as 2 sin^2(theta/2) and derive
                   everything from it. For a solar aperture of 0.2665 deg,
                   1 - cos(theta) is about 1.1e-5, and forming it directly in
                   single precision loses roughly 1% relative accuracy to
                   cancellation (float32 resolves ~6e-8 near 1.0). That error
                   would land in the solid angle, in the cone membership test,
                   and -- because the sampler and the pdf would disagree -- as a
                   systematic bias in the rendered irradiance. The half-angle
                   form has no cancellation. */
                ScalarFloat theta = dr::deg_to_rad(half_aperture);
                ScalarFloat s = dr::sin(0.5f * theta);
                m_sun_one_minus_cos = 2.f * s * s;
                m_sun_solid_angle =
                    2.f * dr::Pi<ScalarFloat> * m_sun_one_minus_cos;

                ScalarVector3f dir = sun_direction_from_props(props);
                if (!dr::all(dr::isfinite(dir)) || !std::isfinite(dr::squared_norm(dir)) ||
                    dr::squared_norm(dir) == 0.f)
                    Throw("sun_direction must be finite and nonzero.");
                m_sun_direction = dr::normalize(dir);
                m_sun_frame     = Frame3f(Vector3f(m_sun_direction));

                m_sun_radiance = has_radiance
                    ? props.get_texture<Texture>("sun_radiance")
                    : props.get_texture<Texture>("sun_irradiance");

                /* Convert irradiance to the radiance of the disk. Getting this
                   backwards is the classic error here, so the two spellings are
                   kept distinct rather than inferred. */
                m_sun_scale = checked_scale(props, "sun_scale");
                if (has_irradiance)
                    m_sun_scale /= dr::Pi<ScalarFloat> * dr::square(dr::sin(theta));
                if (!std::isfinite(m_sun_scale) || !(m_sun_solid_angle > 0.f) ||
                    !std::isfinite(1.f / m_sun_solid_angle))
                    Throw("sun_half_aperture is too small for finite sampling weights.");
                if (m_sun_radiance->is_spatially_varying())
                    Throw("The sun spectrum must be spatially uniform.");
                m_sun_range = m_sun_radiance->wavelength_range();
                if (!dr::all(dr::isfinite(m_sun_range)) || m_sun_range.x() <= 0.f ||
                    m_sun_range.y() <= m_sun_range.x())
                    Throw("The sun spectrum must declare a finite, positive wavelength range.");
                m_sampling_min = dr::minimum(m_sampling_min, m_sun_range.x());
                m_sampling_max = dr::maximum(m_sampling_max, m_sun_range.y());

                /* Fraction of direction samples drawn from the sun cone rather
                   than from the map. This steers variance only: the estimator
                   is unbiased for any value strictly inside (0, 1). It is a
                   tunable rather than something inferred from the spectra,
                   because the relative power would require scalarising a
                   JIT-resident texture mean. Raise it when the sun dominates
                   (short-wave bands), lower it when the sky does (thermal). */
                m_sun_weight =
                    props.get<ScalarFloat>("sun_sampling_weight", 0.25f);
                if (!std::isfinite(m_sun_weight) || !(m_sun_weight > 0.f) || m_sun_weight >= 1.f)
                    Throw("SpectralEnvironmentMapEmitter: "
                          "\"sun_sampling_weight\" must lie strictly between 0 "
                          "and 1 (got %f).", m_sun_weight);
            }

            rebuild_distributions(storage.get());
        }
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("scale", m_scale, ParamFlags::Differentiable);
        cb->put("data", m_texture.tensor(),
                ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("to_world", m_to_world, ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        if (!dr::all(dr::isfinite(m_scale) && m_scale >= 0.f))
            Throw("scale must be finite and non-negative.");
        if (keys.empty() || string::contains(keys, "data")) {
            TensorXf &tensor = m_texture.tensor();
            if (tensor.ndim() != 4 || tensor.shape(0) != m_n_wavelengths ||
                tensor.shape(3) != 1 || tensor.shape(1) < 3 || tensor.shape(2) < 4)
                Throw("data must have shape (wavelength_count, H>=3, W+2>=4, 1); "
                      "the wavelength grid is fixed for this emitter.");
            ScalarVector2u res((uint32_t) tensor.shape(2) - 2, (uint32_t) tensor.shape(1));
            auto host = dr::migrate(tensor.array(), JitBackend::None);
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();
            const ScalarFloat *ptr = host.data();
            for (size_t row = 0; row < (size_t) m_n_wavelengths * res.y(); ++row)
                for (uint32_t x = 1; x <= res.x(); ++x) {
                    ScalarFloat value = ptr[row * (res.x() + 2) + x];
                    if (!std::isfinite(value) || value < 0.f)
                        Throw("data must contain finite, non-negative radiance in its real columns.");
                }
            m_res = res;
            uint32_t sw = res.x() + 2;
            if constexpr (dr::is_jit_v<Float>) {
                const auto &array = tensor.array();
                Float corrected = array;
                UInt32 row = dr::arange<UInt32>((size_t) m_n_wavelengths * res.y()) * sw;
                dr::scatter(corrected, dr::gather<Float>(array, row + res.x()), row);
                dr::scatter(corrected, dr::gather<Float>(array, row + 1u), row + res.x() + 1u);
                m_texture.set_tensor(TensorXf(corrected,
                    { (size_t) m_n_wavelengths, (size_t) res.y(), (size_t) sw, 1 }));
            } else {
                refresh_halo(tensor.array().data(), res, m_n_wavelengths);
                m_texture.update_inplace();
            }
            auto refreshed = dr::migrate(m_texture.tensor().array(), JitBackend::None);
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();
            rebuild_distributions(refreshed.data());
        }
        dr::make_opaque(m_scale);
        Base::parameters_changed(keys);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f scene_sphere = scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(scene_sphere.center, scene_sphere.radius);
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                            m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
        dr::make_opaque(m_bsphere.center, m_bsphere.radius);
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Vector3f v = m_to_world->eval(si.time).inverse() * (-si.wi);
        return depolarizer<Spectrum>(
            eval_spectrum(direction_to_uv(v), si.wavelengths, active));
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2,
                                          const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        Point2f offset = warp::square_to_uniform_disk_concentric(sample2);

        auto [d_local, uv] = sample_local_direction(sample3, active, true);
        Float pdf = mixture_pdf_local(d_local, true);
        active &= pdf > 0.f;

        Vector3f d_global = m_to_world->eval(time) * -d_local;

        Vector3f perpendicular_offset =
            Frame3f(d_global).to_world(Vector3f(offset.x(), offset.y(), 0));
        Point3f origin =
            m_bsphere.center + (perpendicular_offset - d_global) * m_bsphere.radius;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.t    = 0.f;
        si.time = time;
        si.p    = origin;
        si.uv   = uv;
        auto [wavelengths, weight] =
            sample_wavelengths(si, wavelength_sample, active);

        Float r2 = dr::square(m_bsphere.radius);
        Ray3f ray(origin, d_global, time, wavelengths);
        weight *= dr::Pi<Float> * r2 / pdf;

        return { ray, weight & active };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        auto [d_local, uv] = sample_local_direction(sample, active);
        Float pdf = mixture_pdf_local(d_local);
        active &= pdf > 0.f;

        Float radius =
            dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center));
        Float dist = 2.f * radius;

        Vector3f d = m_to_world->eval(it.time) * d_local;

        DirectionSample3f ds;
        ds.p    = it.p + d * dist;
        ds.n    = -d;
        ds.uv   = uv;
        ds.time = it.time;
        ds.pdf     = dr::select(active, pdf, 0.f);
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        auto weight =
            depolarizer<Spectrum>(eval_spectrum(uv, it.wavelengths, active)) /
            ds.pdf;

        return { ds, weight & active };
    }

    Float pdf_direction(const Interaction3f & /*it*/,
                        const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f d = m_to_world->eval(ds.time).inverse() * ds.d;
        return mixture_pdf_local(d);
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        return depolarizer<Spectrum>(eval_spectrum(ds.uv, it.wavelengths, active));
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            // Keep the data-driven proposal, with uniform support over the
            // union of the sky and sun ranges. This also covers sun-only bands
            // and zero-valued map intervals without relying on Texture::pdf_spectrum.
            Wavelength u = math::sample_shifted<Wavelength>(sample);
            ScalarFloat uniform_weight = m_has_sun ? .5f : .01f;
            auto choose_uniform = u < uniform_weight;
            Wavelength u_map = dr::clip((u - uniform_weight) / (1.f - uniform_weight), 0.f, 1.f);
            auto [lambda_map, unused_pdf] = m_wavelength_distr.sample_pdf(u_map, active);
            Wavelength lambda_uniform = dr::fmadd(u / uniform_weight,
                m_sampling_max - m_sampling_min, m_sampling_min);
            Wavelength wavelengths = dr::select(choose_uniform, lambda_uniform, lambda_map);
            Wavelength pdf = (1.f - uniform_weight) *
                m_wavelength_distr.eval_pdf_normalized(wavelengths, active) +
                uniform_weight / (m_sampling_max - m_sampling_min);
            Spectrum value = depolarizer<Spectrum>(eval_spectrum(si.uv, wavelengths, active));
            return { wavelengths, (value / pdf) & active };
        } else {
            DRJIT_MARK_USED(si);
            DRJIT_MARK_USED(sample);
            DRJIT_MARK_USED(active);
            return { dr::zeros<Wavelength>(), dr::zeros<Spectrum>() };
        }
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            return { dr::zeros<PositionSample3f>(), dr::NaN<Float> };
        } else {
            NotImplementedError("sample_position");
        }
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SpectralEnvironmentMapEmitter[" << std::endl;
        if (!m_filename.empty())
            oss << "  filename = \"" << m_filename << "\"," << std::endl;
        oss << "  res = " << m_res << "," << std::endl
            << "  wavelengths = " << m_n_wavelengths << " x " << m_wavelength_step
            << "nm in [" << m_wavelength_min << ", " << m_wavelength_max << "],"
            << std::endl
            << "  scale = " << m_scale << "," << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(SpectralEnvironmentMapEmitter)

protected:
    static ScalarFloat checked_scale(const Properties &props, const char *name) {
        ScalarFloat value = props.get<ScalarFloat>(name, 1.f);
        if (!std::isfinite(value) || value < 0.f)
            Throw("%s must be finite and non-negative.", name);
        return value;
    }

    /// Row stride of the phi axis, including one halo column on each side.
    uint32_t res_stride() const { return m_res.x() + 2; }

    /// Flat index into the (wavelength, theta, phi) storage buffer.
    size_t flat(uint32_t w, uint32_t y, uint32_t x) const {
        return ((size_t) w * m_res.y() + y) * res_stride() + x;
    }

    ScalarFloat half_texel() const { return .5f / m_res.x(); }

    /// Resolve the direction *toward* the sun from the scene, in local frame.
    static ScalarVector3f sun_direction_from_props(const Properties &props) {
        bool has_vec = props.has_property("sun_direction");
        bool has_ang = props.has_property("sun_elevation") ||
                       props.has_property("sun_azimuth");
        if (has_vec && has_ang)
            Throw("SpectralEnvironmentMapEmitter: specify either "
                  "\"sun_direction\" or \"sun_elevation\"/\"sun_azimuth\", not "
                  "both.");
        if (has_vec)
            return props.get<ScalarVector3f>("sun_direction");
        if (!has_ang)
            Throw("SpectralEnvironmentMapEmitter: a sun spectrum was given, so "
                  "its position is required -- set \"sun_direction\" (pointing "
                  "from the scene toward the sun) or "
                  "\"sun_elevation\"/\"sun_azimuth\" in degrees.");

        /* Same convention as the map itself: +Y is up, azimuth is measured
           from -Z toward +X, matching uv_to_direction(). */
        ScalarFloat el = dr::deg_to_rad(
                        props.get<ScalarFloat>("sun_elevation", 45.f)),
                    az = dr::deg_to_rad(
                        props.get<ScalarFloat>("sun_azimuth", 0.f));
        ScalarFloat ce = dr::cos(el), se = dr::sin(el);
        return ScalarVector3f(dr::sin(az) * ce, se, -dr::cos(az) * ce);
    }

    /// Solid-angle density of the sky (map) sampling strategy, local direction.
    Float sky_pdf_local(const Vector3f &d, bool emission = false) const {
        Point2f uv = direction_to_uv(d);
        uv.x() -= half_texel();
        uv -= dr::floor(uv);
        Float inv_sin_theta = dr::safe_rsqrt(dr::maximum(
            dr::square(d.x()) + dr::square(d.z()),
            dr::square(dr::Epsilon<Float>)));
        const Warp &warp = emission && m_mis_compensation ? m_emission_warp : m_warp;
        return warp.eval(uv) * inv_sin_theta *
               (1.f / (2.f * dr::square(dr::Pi<Float>)));
    }

    /** \brief Is a (unit) local direction inside the sun's cone?
     *
     * Tested via the chord length rather than ``dot(d, sun) >= cos(theta)``.
     * Both vectors are unit, so ``|d - sun|^2 = 2 (1 - dot)``; the component
     * differences are exact for nearby directions (Sterbenz), whereas
     * ``1 - dot`` cancels catastrophically for a cone this narrow.
     */
    Mask sun_inside(const Vector3f &d) const {
        return dr::squared_norm(d - Vector3f(m_sun_direction)) <=
               2.f * m_sun_one_minus_cos;
    }

    /// Solid-angle density of the sun-cone strategy: uniform inside, 0 outside.
    Float sun_pdf_local(const Vector3f &d) const {
        if (!m_has_sun)
            return 0.f;
        return dr::select(sun_inside(d), 1.f / m_sun_solid_angle, 0.f);
    }

    /** \brief Uniform cone sampling parameterised by ``1 - cos(theta)``.
     *
     * Equivalent to \ref warp::square_to_uniform_cone, but that helper takes
     * ``cos_cutoff`` and reconstructs ``1 - cos_cutoff`` internally, which
     * reintroduces the cancellation this class is at pains to avoid. Keeping
     * the sampler and the pdf derived from the same stable quantity is what
     * makes the estimator unbiased.
     */
    Vector3f sample_sun_cone(const Point2f &sample) const {
        Float omc(m_sun_one_minus_cos);
        Point2f p = warp::square_to_uniform_disk_concentric(sample);
        Float pn  = dr::squared_norm(p);
        Float z   = 1.f - omc * pn;
        p *= dr::safe_sqrt(omc * (2.f - omc * pn));
        return Vector3f(p.x(), p.y(), z);
    }

    /** \brief Density of the combined strategy.
     *
     * Directions are drawn from the map's warp with probability
     * ``1 - m_sun_weight`` and from the sun cone with probability
     * ``m_sun_weight``; the estimator is unbiased for any weight in (0, 1), so
     * the heuristic used to pick it only affects variance.
     */
    Float mixture_pdf_local(const Vector3f &d, bool emission = false) const {
        if (!m_has_sun)
            return sky_pdf_local(d, emission);
        return m_sun_weight * sun_pdf_local(d) +
               (1.f - m_sun_weight) * sky_pdf_local(d, emission);
    }

    /// Draw a local direction from the combined strategy; also returns its uv.
    std::pair<Vector3f, Point2f>
    sample_local_direction(const Point2f &sample, Mask active, bool emission = false) const {
        const Warp &warp = emission && m_mis_compensation ? m_emission_warp : m_warp;
        if (!m_has_sun) {
            auto [uv, pdf] = warp.sample(sample, nullptr, active);
            uv.x() += half_texel();
            Float inv_sin_theta;
            Vector3f d = uv_to_direction(uv, inv_sin_theta);
            return { d, uv };
        }

        Mask pick_sun = sample.x() < m_sun_weight;

        // Reuse the first sample dimension for the strategy choice.
        Float s_sun = sample.x() / dr::maximum(m_sun_weight, 1e-9f),
              s_sky = (sample.x() - m_sun_weight) /
                      dr::maximum(1.f - m_sun_weight, 1e-9f);

        auto [uv_sky, pdf_sky] =
            warp.sample(Point2f(dr::clip(s_sky, 0.f, 1.f), sample.y()),
                          nullptr, active);
        uv_sky.x() += half_texel();
        Float inv_sin_theta;
        Vector3f d_sky = uv_to_direction(uv_sky, inv_sin_theta);

        Vector3f local_cone =
            sample_sun_cone(Point2f(dr::clip(s_sun, 0.f, 1.f), sample.y()));
        Vector3f d_sun = m_sun_frame.to_world(local_cone);

        Vector3f d = dr::select(pick_sun, d_sun, d_sky);
        return { d, direction_to_uv(d) };
    }

    Vector3f uv_to_direction(const Point2f &uv, Float &inv_sin_theta) const {
        Float theta = uv.y() * dr::Pi<Float>,
              phi   = uv.x() * dr::TwoPi<Float>;
        auto [sin_theta, cos_theta] = dr::sincos(theta);
        auto [sin_phi, cos_phi]     = dr::sincos(phi);
        inv_sin_theta = dr::rcp(dr::maximum(sin_theta, dr::Epsilon<Float>));
        return Vector3f(sin_phi * sin_theta, cos_theta, -cos_phi * sin_theta);
    }

    Point2f direction_to_uv(const Vector3f &d) const {
        return Point2f(dr::atan2(d.x(), -d.z()) * dr::InvTwoPi<Float>,
                       dr::safe_acos(d.y()) * dr::InvPi<Float>);
    }

    static void refresh_halo(ScalarFloat *ptr, const ScalarVector2u &res,
                             uint32_t n_wavelengths) {
        const uint32_t sw = res.x() + 2;
        for (uint32_t w = 0; w < n_wavelengths; ++w) {
            for (uint32_t y = 0; y < res.y(); ++y) {
                ScalarFloat *row =
                    ptr + ((size_t) w * res.y() + y) * sw;
                row[0]           = row[res.x()];
                row[res.x() + 1] = row[1];
            }
        }
    }

    /// Build the direction-sampling warp and the wavelength distribution.
    void rebuild_distributions(const ScalarFloat *data) {
        const ScalarVector2u warp_res(m_res.x() + 1, m_res.y());
        const size_t n = (size_t) warp_res.x() * warp_res.y();

        std::unique_ptr<ScalarFloat[]> weight(new ScalarFloat[n]);
        std::vector<ScalarFloat> mean_spectrum(m_n_wavelengths, 0.f);

        ScalarFloat maximum = 0.f;
        for (uint32_t w = 0; w < m_n_wavelengths; ++w)
            for (uint32_t y = 0; y < m_res.y(); ++y)
                for (uint32_t x = 1; x <= m_res.x(); ++x)
                    maximum = dr::maximum(maximum, data[flat(w, y, x)]);
        double normalization = maximum > 0.f ? 1.0 / double(maximum) : 1.0;

        /* Direction weights are the band-integrated radiance, integral L dlambda.
           `envmap` weights by photometric luminance here, which is meaningless
           outside the visible range. */
        for (uint32_t y = 0; y < warp_res.y(); ++y) {
            for (uint32_t x = 0; x < warp_res.x(); ++x) {
                ScalarFloat sum = 0.f;
                for (uint32_t w = 0; w < m_n_wavelengths; ++w)
                    sum += ScalarFloat(double(data[flat(w, y, x + 1)]) * normalization) *
                           (w == 0 || w + 1 == m_n_wavelengths ? .5f : 1.f) /
                           (m_n_wavelengths - 1);
                weight[(size_t) y * warp_res.x() + x] = sum;
            }
        }

        if (m_mis_compensation) {
            // Light tracing has no complementary BSDF strategy at emission.
            // Build its proposal before subtracting the MIS compensation term.
            std::vector<ScalarFloat> emission_weights(weight.get(), weight.get() + n);
            ScalarFloat sum = 0.f;
            for (uint32_t y = 0; y < warp_res.y(); ++y) {
                ScalarFloat sine = dr::maximum(dr::sin(y * dr::Pi<ScalarFloat> /
                    (warp_res.y() - 1)), dr::Epsilon<ScalarFloat>);
                for (uint32_t x = 0; x < warp_res.x(); ++x) {
                    ScalarFloat &v = emission_weights[(size_t) y * warp_res.x() + x];
                    v *= sine;
                    sum += v;
                }
            }
            if (!(sum > 0.f))
                std::fill(emission_weights.begin(), emission_weights.end(), 1.f);
            m_emission_warp = Warp(emission_weights.data(), warp_res);
        }

        ScalarFloat offset = 0.f;
        if (m_mis_compensation) {
            ScalarFloat min_w = dr::Infinity<ScalarFloat>;
            double accum = 0.0;
            for (uint32_t y = 0; y < warp_res.y(); ++y) {
                for (uint32_t x = 0; x < warp_res.x() - 1u; ++x) {
                    ScalarFloat v = weight[(size_t) y * warp_res.x() + x];
                    min_w = dr::minimum(min_w, v);
                    accum += (double) v;
                }
            }
            offset = ScalarFloat(accum /
                                 ((warp_res.x() - 1u) * (size_t) warp_res.y()));
            if (offset - min_w <= 0.01f * offset)
                offset = 0.f;
        }

        /* The solid angle of an equirectangular texel shrinks as sin(theta)
           toward the poles; without this the warp over-samples them. */
        ScalarFloat theta_scale = 1.f / (warp_res.y() - 1) * dr::Pi<ScalarFloat>;
        for (uint32_t y = 0; y < warp_res.y(); ++y) {
            ScalarFloat sin_theta = dr::maximum(dr::sin(y * theta_scale), dr::Epsilon<ScalarFloat>);
            for (uint32_t x = 0; x < warp_res.x(); ++x) {
                ScalarFloat &v = weight[(size_t) y * warp_res.x() + x];
                v = dr::maximum(v - offset, 0.f) * sin_theta;
            }
            // Accumulate the sin(theta)-weighted mean spectrum in the same pass.
            for (uint32_t w = 0; w < m_n_wavelengths; ++w) {
                ScalarFloat row_sum = 0.f;
                for (uint32_t x = 0; x < m_res.x(); ++x)
                    row_sum += ScalarFloat(double(data[flat(w, y, x + 1)]) * normalization) / m_res.x();
                mean_spectrum[w] += row_sum * sin_theta / m_res.y();
            }
        }

        ScalarFloat max_weight = 0.f;
        for (size_t i = 0; i < n; ++i)
            max_weight = dr::maximum(max_weight, weight[i]);
        // A dark sky still needs a valid proposal (e.g. for a sun-only emitter).
        if (!(max_weight > 0.f))
            std::fill_n(weight.get(), n, 1.f);
        m_warp = Warp(weight.get(), warp_res);

        /* A constant map would give a uniform distribution, which is correct.
           Guard against an all-zero map, which has no valid density. */
        ScalarFloat total = 0.f;
        for (ScalarFloat v : mean_spectrum)
            total += v;
        if (!(total > 0.f))
            std::fill(mean_spectrum.begin(), mean_spectrum.end(), 1.f);

        if constexpr (is_spectral_v<Spectrum>) {
            m_wavelength_distr = ContinuousDistribution<Wavelength>(
                ScalarVector2f(m_wavelength_min, m_wavelength_max),
                mean_spectrum.data(), mean_spectrum.size());
        }

    }

    UnpolarizedSpectrum eval_spectrum(Point2f uv, const Wavelength &wavelengths,
                                      Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Vector2f res(m_res);

            /* Texel-centered phi over the W real columns (offset by one to skip
               the leading halo column), align-corners theta over the H rows.
               The texture's own ``pos * res - 0.5`` absorbs the half-texel
               shift, and WrapMode::Clamp plus the halo give periodic phi. */
            Float u = uv.x() - dr::floor(uv.x()),
                  v = dr::clip(uv.y(), 0.f, 1.f);
            Float pos_x = dr::fmadd(u, res.x(), 1.f) / (res.x() + 2.f),
                  pos_y = dr::fmadd(v, res.y() - 1.f, 0.5f) / res.y();

            UnpolarizedSpectrum result = dr::zeros<UnpolarizedSpectrum>();

            for (size_t i = 0; i < dr::size_v<UnpolarizedSpectrum>; ++i) {
                Float lambda = wavelengths[i];

                /* Outside the tabulated range the answer is zero, not the
                   nearest band: clamping would fabricate radiance at
                   wavelengths that were never measured. */
                Mask in_range = active && lambda >= m_wavelength_min &&
                                lambda <= m_wavelength_max;

                Float t = (lambda - m_wavelength_min) / m_wavelength_step;
                Float pos_z = (t + 0.5f) / (Float) m_n_wavelengths;

                dr::Array<Float, 3> pos(pos_x, pos_y, pos_z);
                Texel texel =
                    m_texture.template eval<Texel>(pos, in_range);

                result[i] = dr::select(in_range, texel.x() * m_scale, 0.f);
            }

            /* Add the analytic sun disk. The direction is recovered from uv,
               which is a bijection with the sphere, so every caller that
               already works in uv picks the sun up for free -- including rays
               that escape the scene, which is what makes the disk visible to
               the sensor rather than merely an illumination source. */
            if (m_has_sun) {
                Float unused;
                Vector3f d = uv_to_direction(Point2f(u, v), unused);
                Mask inside = active && sun_inside(d);

                SurfaceInteraction3f si_sun = dr::zeros<SurfaceInteraction3f>();
                si_sun.wavelengths = wavelengths;
                UnpolarizedSpectrum sun =
                    m_sun_radiance->eval(si_sun, inside) * m_sun_scale;

                // A texture's declared range is its finite emission support,
                // including uniform textures whose eval() otherwise extrapolates.
                result += dr::select(inside && wavelengths >= m_sun_range.x() &&
                                     wavelengths <= m_sun_range.y(), sun, 0.f);
            }

            return result;
        } else {
            DRJIT_MARK_USED(uv);
            DRJIT_MARK_USED(wavelengths);
            DRJIT_MARK_USED(active);
            return dr::zeros<UnpolarizedSpectrum>();
        }
    }

protected:
    std::string m_filename;
    BoundingSphere3f m_bsphere;
    Tex m_texture;
    ScalarVector2u m_res;
    Warp m_warp, m_emission_warp;
    ContinuousDistribution<Wavelength> m_wavelength_distr;
    Float m_scale;
    ScalarFloat m_sampling_min = 0.f, m_sampling_max = 0.f;
    ScalarVector2f m_sun_range = ScalarVector2f(0.f);
    ScalarFloat m_wavelength_min = 0.f;
    ScalarFloat m_wavelength_max = 0.f;
    ScalarFloat m_wavelength_step = 0.f;
    uint32_t m_n_wavelengths = 0;
    bool m_mis_compensation = false;

    // Analytic sun disk (optional)
    bool m_has_sun = false;
    ref<Texture> m_sun_radiance;
    ScalarVector3f m_sun_direction = ScalarVector3f(0.f, 1.f, 0.f);
    Frame3f m_sun_frame;
    /// 1 - cos(half aperture), stored as 2 sin^2(theta/2).
    ScalarFloat m_sun_one_minus_cos = 0.f;
    ScalarFloat m_sun_solid_angle = 0.f;
    ScalarFloat m_sun_scale = 1.f;
    /// Probability of drawing a direction from the sun cone rather than the map.
    ScalarFloat m_sun_weight = 0.f;

    MI_TRAVERSE_CB(Base, m_bsphere, m_texture, m_warp, m_emission_warp, m_wavelength_distr,
                   m_scale, m_sun_radiance, m_sun_frame)
};

MI_EXPORT_PLUGIN(SpectralEnvironmentMapEmitter)
NAMESPACE_END(mitsuba)
