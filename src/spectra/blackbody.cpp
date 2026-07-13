#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>

#include <algorithm>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-blackbody:

Blackbody spectrum (:monosp:`blackbody`)
----------------------------------------

.. pluginparameters::

 * - wavelength_min
   - |float|
   - Minimum wavelength of the spectral range in nanometers. (Default: 360nm)

 * - wavelength_max
   - |float|
   - Maximum wavelength of the spectral range in nanometers. (Default: 830nm)

 * - temperature
   - |float| or |texture|
   - Black body temperature in Kelvins. Can be a constant or a texture (e.g. bitmap).
   - |exposed|

 * - temperature_attribute
   - |string|
   - Name of a per-vertex scalar attribute (e.g. from a PLY) to use as temperature in Kelvins.
   - If specified, this bypasses the texture system and queries the shape attribute at the shading point.

 * - emissivity
   - |float| or |spectrum|
   - Homogeneous surface emissivity. A float selects an explicit greybody fast path.
   - A wavelength-dependent curve can be supplied inline or loaded from an ``.spd`` file
   - using Mitsuba's standard spectrum syntax. Curve values are clamped to :math:`[0, 1]`
   - and the curve must cover the complete ``wavelength_min`` to ``wavelength_max`` interval.
   - (Default: 1.0)
   - |exposed|, |differentiable|

 * - sampling_temperature
   - |float|
   - Representative temperature (Kelvins) used by summary quantities such as
   - :monosp:`mean()` and :monosp:`max()`. Wavelength sampling uses the local temperature.
   - Default: if :monosp:`temperature` is provided as a constant, that value; otherwise 300 K.

This plugin models blackbody or greybody radiation for a specified temperature.
For spectral radiance :math:`B_\lambda(T)` and emissivity
:math:`\epsilon_\lambda`, the emitted radiance is

.. math::

    L_\lambda(T) = \epsilon_\lambda B_\lambda(T), \qquad
    0 \leq \epsilon_\lambda \leq 1.

A scalar ``emissivity`` is evaluated outside the wavelength-dependent path. A
spectral curve remains wavelength-dependent even when its tabulated samples
happen to be equal; there is intentionally no automatic flat-curve detection.
The blackbody wavelength sampling distribution is not changed by emissivity.
Instead, the Monte Carlo weight is multiplied by emissivity at the sampled
wavelength, which is unbiased as long as the curve covers the configured
wavelength interval.

See :ref:`thermal-emission` for the complete radiometric model, conservation
relationships, and sampling derivation.

.. tabs::
    .. code-tab:: xml
        :name: blackbody-emissivity

        <spectrum type="blackbody" name="radiance">
            <float name="temperature" value="5000"/>
            <spectrum name="emissivity" filename="materials/aluminum.spd"/>
        </spectrum>

    .. code-tab:: python

        'radiance': {
            'type': 'blackbody',
            'temperature': 5000,
            'emissivity': {
                'type': 'spectrum',
                'filename': 'materials/aluminum.spd'
            }
        }

This spectrum type only makes sense for specifying emission and is unavailable
in non-spectral rendering modes.

Note that attaching a black body spectrum to the intensity property of a emitter introduces
physical units into the rendering process of Mitsuba 3, which is ordinarily a unitless system.
Specifically, the black body spectrum has units of power (:math:`W`) per unit area (:math:`m^{-2}`)
per steradian (:math:`sr^{-1}`) per unit wavelength (:math:`nm^{-1}`). As a consequence,
your scene should be modeled in meters for this plugin to work properly.

 */

template <typename Float, typename Spectrum>
class BlackBodySpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    // A few natural constants
    constexpr static ScalarFloat c = ScalarFloat(2.99792458e+8);   /// Speed of light
    constexpr static ScalarFloat h = ScalarFloat(6.62607004e-34);  /// Planck constant
    constexpr static ScalarFloat k = ScalarFloat(1.38064852e-23);  /// Boltzmann constant
    constexpr static ScalarFloat b = ScalarFloat(2.89777196e-3);   /// Wien displacement constant

    /// First and second radiation static constants
    constexpr static ScalarFloat c0 = 2 * h * c * c;
    constexpr static ScalarFloat c1 = h * c / k;

    BlackBodySpectrum(const Properties &props) : Texture(props) {
        m_wavelength_range = ScalarVector2f(
            props.get<ScalarFloat>("wavelength_min", MI_CIE_MIN),
            props.get<ScalarFloat>("wavelength_max", MI_CIE_MAX)
        );

        if (!std::isfinite(m_wavelength_range.x()) ||
            !std::isfinite(m_wavelength_range.y()) ||
            m_wavelength_range.x() >= m_wavelength_range.y()) {
            Throw("Blackbody wavelength range must be finite and increasing, got [%f, %f].",
                  m_wavelength_range.x(), m_wavelength_range.y());
        }

        if (!props.has_property("emissivity")) {
            m_emissivity_constant = dr::opaque<Float>(1.f);
        } else if (props.type("emissivity") == Properties::Type::Float ||
                   props.type("emissivity") == Properties::Type::Integer) {
            ScalarFloat emissivity = props.get<ScalarFloat>("emissivity");
            if (!std::isfinite(emissivity))
                Throw("Blackbody emissivity must be finite, got %f.", emissivity);
            m_emissivity_constant =
                dr::opaque<Float>(dr::clip(emissivity, ScalarFloat(0.f), ScalarFloat(1.f)));
        } else if (props.type("emissivity") == Properties::Type::Spectrum ||
                   props.type("emissivity") == Properties::Type::Object) {
            if (props.type("emissivity") == Properties::Type::Spectrum) {
                Properties::Spectrum emissivity =
                    props.get<Properties::Spectrum>("emissivity");
                validate_emissivity_spectrum(emissivity);

                // ContinuousDistribution rejects negative entries before eval()
                // can clamp them, so physically bound raw inline/file samples
                // before constructing the regular or irregular spectrum plugin.
                for (double &value : emissivity.values)
                    value = std::min(std::max(value, 0.0), 1.0);

                Properties emissivity_props(
                    emissivity.is_regular() ? "regular" : "irregular");
                emissivity_props.set("value", std::move(emissivity));
                m_emissivity_tex =
                    PluginManager::instance()->create_object<Texture>(emissivity_props);
            } else {
                m_emissivity_tex = props.get_texture<Texture>("emissivity");
            }

            m_emissivity_is_constant = false;
            validate_emissivity_texture();
        } else {
            Throw("Blackbody emissivity must be a float or spectrum, got %s.",
                  property_type_name(props.type("emissivity")));
        }

        m_use_attribute = props.has_property("temperature_attribute");
        if (m_use_attribute) {
            m_temperature_attribute = props.get<std::string>("temperature_attribute");

            // Default sampling temperature when using spatially varying attribute
            m_sampling_temperature =
                props.get<ScalarFloat>("sampling_temperature", ScalarFloat(300.0));
        } else {
            // Default base temperature if not specified, or specified as a texture object
            ScalarFloat default_temp = ScalarFloat(300.0);

            // Only read "temperature" as a scalar if it is a scalar
            if (props.has_property("temperature") &&
                (props.type("temperature") == Properties::Type::Float ||
                 props.type("temperature") == Properties::Type::Integer)) {
                default_temp = props.get<ScalarFloat>("temperature");
            }

            // This accepts either a float (constant texture) or a texture object (bitmap/etc.)
            m_temperature_tex = props.get_texture<Texture>("temperature", default_temp);

            // Default sampling temp: use provided scalar temperature if present, else 300 K
            m_sampling_temperature =
                props.get<ScalarFloat>("sampling_temperature", default_temp);
        }

        parameters_changed();
    }

    void traverse(TraversalCallback *callback) override {
        // Only expose the temperature object when it is texture-backed.
        if (!m_use_attribute)
            callback->put("temperature", m_temperature_tex, ParamFlags::Differentiable);

        callback->put("sampling_temperature", m_sampling_temperature,
                      ParamFlags::NonDifferentiable);

        if (m_emissivity_is_constant)
            callback->put("emissivity", m_emissivity_constant,
                          ParamFlags::Differentiable);
        else
            callback->put("emissivity", m_emissivity_tex,
                          ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        if (m_emissivity_is_constant) {
            if constexpr (dr::is_jit_v<Float>) {
                if (unlikely(m_emissivity_constant.size() != 1))
                    Throw("Updated blackbody emissivity with a float of size %d",
                          m_emissivity_constant.size());
            }
            m_emissivity_constant = dr::clip(m_emissivity_constant, 0.f, 1.f);
            dr::make_opaque(m_emissivity_constant);
        } else {
            validate_emissivity_texture();
        }

        std::tie(m_integral_min, m_integral) =
            integral_bounds(ScalarFloat(m_sampling_temperature));
    }

    /// Evaluate Planck's law for the provided temperature (Kelvins).
    UnpolarizedSpectrum eval_impl(const Wavelength &wavelengths,
                                  UnpolarizedSpectrum temp_K,
                                  Mask active_) const {
        if constexpr (is_spectral_v<Spectrum>) {
            /* The scale factors of 1e-9f are needed to perform a conversion between
               densities per unit nanometer and per unit meter. */
            Wavelength lambda  = wavelengths * 1e-9f,
                       lambda2 = dr::square(lambda),
                       lambda5 = dr::square(lambda2) * lambda;

            dr::mask_t<Wavelength> active = active_;
            active &= wavelengths >= m_wavelength_range.x()
                   && wavelengths <= m_wavelength_range.y();

            // Avoid pathological values (<= 0 K) producing NaNs/Infs
            temp_K = dr::maximum(temp_K, UnpolarizedSpectrum(Float(1e-3f)));

            /* Watts per unit surface area (m^-2)
                     per unit wavelength (nm^-1)
                     per unit steradian (sr^-1) */
            UnpolarizedSpectrum P = 1e-9f * c0 / (lambda5 *
                    (dr::exp(c1 / (lambda * temp_K)) - 1.f));

            return P & active;
        } else {
            DRJIT_MARK_USED(wavelengths);
            DRJIT_MARK_USED(temp_K);
            DRJIT_MARK_USED(active_);
            Throw("Not implemented for non-spectral modes");
        }
    }

    /// Compute the temperature at the shading point as an UnpolarizedSpectrum (broadcast).
    UnpolarizedSpectrum temperature_at_si(const SurfaceInteraction3f &si,
                                      Mask active) const {
        if (!m_use_attribute) {
            // Texture path (constant/bitmap/etc.) -> MUST be scalar temperature
            Float t = m_temperature_tex->eval_1(si, active);
            return UnpolarizedSpectrum(t);
        }

        // Attribute path ...
        if constexpr (dr::is_jit_v<Float>) {
            Mask valid = active && si.is_valid();
            Float t = si.shape->eval_attribute_1(m_temperature_attribute, si, valid);
            return UnpolarizedSpectrum(t);
        } else {
            if (!active || !si.is_valid() || !si.shape)
                return dr::zeros<UnpolarizedSpectrum>();
            Float t = si.shape->eval_attribute_1(m_temperature_attribute, si, active);
            return UnpolarizedSpectrum(t);
        }
    }

    /// Evaluate and physically bound homogeneous spectral surface emissivity.
    UnpolarizedSpectrum emissivity_at_si(const SurfaceInteraction3f &si,
                                         Mask active) const {
        return dr::clip(m_emissivity_tex->eval(si, active), 0.f, 1.f);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        UnpolarizedSpectrum temp_K = temperature_at_si(si, active);
        UnpolarizedSpectrum value = eval_impl(si.wavelengths, temp_K, active);
        if (m_emissivity_is_constant)
            return value * m_emissivity_constant;
        return value * emissivity_at_si(si, active);
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f &si, Mask active_) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength lambda  = si.wavelengths * 1e-9f,
                       lambda2 = dr::square(lambda),
                       lambda5 = dr::square(lambda2) * lambda;

            dr::mask_t<Wavelength> active = active_;
            active &= si.wavelengths >= m_wavelength_range.x()
                   && si.wavelengths <= m_wavelength_range.y();

            Wavelength K = dr::maximum(Wavelength(temperature_at_si(si, active_)),
                                       Wavelength(1e-3f));
            auto [integral_min, integral] = integral_bounds(K);

            DRJIT_MARK_USED(integral_min);

            // Wien's approximation to Planck's law at the local temperature.
            Wavelength pdf = 1e-9f * c0 * dr::exp(-c1 / (lambda * K))
                / (lambda5 * integral);

            return pdf & active;
        } else {
            DRJIT_MARK_USED(si);
            DRJIT_MARK_USED(active_);
            Throw("Not implemented for non-spectral modes");
        }
    }

    template <typename Value>
    std::pair<Value, Value> cdf_and_pdf(Value lambda, Value K) const {
        Value c1_2 = dr::square(c1),
              c1_3 = c1_2 * c1,
              c1_4 = dr::square(c1_2);

        K = dr::maximum(K, Value(1e-3f));

        Value K2 = dr::square(K),
              K3 = K2 * K;

        lambda *= 1e-9f;

        Value lambda2 = dr::square(lambda),
              lambda3 = lambda2 * lambda,
              lambda5 = lambda2 * lambda3;

        Value expval = dr::exp(-c1 / (K * lambda));

        Value cdf = c0 * K * expval *
                (c1_3 + 3 * c1_2 * K * lambda + 6 * c1 * K2 * lambda2 +
                 6 * K3 * lambda3) / (c1_4 * lambda3);

        Value pdf = 1e-9f * c0 * expval / lambda5;

        return { cdf, pdf };
    }

    template <typename Value>
    std::pair<Value, Value> integral_bounds(Value K) const {
        Value integral_min = cdf_and_pdf(Value(m_wavelength_range.x()), K).first,
              integral_max = cdf_and_pdf(Value(m_wavelength_range.y()), K).first;

        return { integral_min, integral_max - integral_min };
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f &si,
                    const Wavelength &sample_, Mask active_) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active_);

        using WavelengthMask = dr::mask_t<Wavelength>;

        if constexpr (is_spectral_v<Spectrum>) {
            WavelengthMask active = active_;
            Wavelength K = dr::maximum(Wavelength(temperature_at_si(si, active_)),
                                       Wavelength(1e-3f));
            auto [integral_min, integral] = integral_bounds(K);

            Wavelength sample = dr::fmadd(sample_, integral, integral_min);

            const ScalarFloat eps        = 1e-5f,
                              eps_domain = eps * (m_wavelength_range.y() - m_wavelength_range.x());
            Wavelength eps_value = eps * integral;

            Wavelength a = m_wavelength_range.x(),
                       b = m_wavelength_range.y(),
                       t = .5f * (m_wavelength_range.x() + m_wavelength_range.y()),
                       value, deriv;

            do {
                // Fall back to a bisection step when t is out of bounds
                WavelengthMask bisect_mask = !((t > a) && (t < b));
                dr::masked(t, bisect_mask && active) = .5f * (a + b);

                // Evaluate the definite integral and its derivative (i.e. the spline)
                std::tie(value, deriv) = cdf_and_pdf(t, K);
                value -= sample;

                // Update which lanes are still active
                active = active && (dr::abs(value) > eps_value) && (b - a > eps_domain);

                if (dr::none_nested(active))
                    break;

                // Update the bisection bounds
                WavelengthMask update_mask = value <= 0.f;
                dr::masked(a,  update_mask) = t;
                dr::masked(b, !update_mask) = t;

                // Perform a Newton step
                dr::masked(t, active) = t - value / deriv;
            } while (true);

            Wavelength pdf = deriv / integral;
            SurfaceInteraction3f si_sampled(si);
            si_sampled.wavelengths = t;

            UnpolarizedSpectrum val =
                eval_impl(t, UnpolarizedSpectrum(K), active_);
            if (m_emissivity_is_constant)
                val *= m_emissivity_constant;
            else
                val *= emissivity_at_si(si_sampled, active_);

            return { t, val / pdf };
        } else {
            DRJIT_MARK_USED(sample_);
            Throw("Not implemented for non-spectral modes");
        }
    }

    Float mean() const override {
        ScalarFloat width = m_wavelength_range.y() - m_wavelength_range.x();
        if (m_emissivity_is_constant)
            return m_emissivity_constant * m_integral / width;

        // Simpson quadrature is only used for the infrequently queried mean().
        // Rendering still evaluates the original spectral curve directly.
        constexpr size_t interval_count = 64;
        ScalarFloat step = width / ScalarFloat(interval_count);
        Float integral = 0.f;
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        for (size_t i = 0; i <= interval_count; ++i) {
            ScalarFloat wavelength = m_wavelength_range.x() + ScalarFloat(i) * step;
            si.wavelengths = Wavelength(wavelength);

            UnpolarizedSpectrum value =
                eval_impl(si.wavelengths,
                          UnpolarizedSpectrum(m_sampling_temperature), true) *
                emissivity_at_si(si, true);
            ScalarFloat coefficient =
                (i == 0 || i == interval_count) ? ScalarFloat(1.f) :
                (i & 1) ? ScalarFloat(4.f) : ScalarFloat(2.f);
            integral += coefficient * dr::mean(value);
        }

        return integral * step / (ScalarFloat(3.f) * width);
    }

    ScalarVector2f wavelength_range() const override {
        return m_wavelength_range;
    }

    ScalarFloat spectral_resolution() const override {
        return 0.f;
    }

    ScalarFloat max() const override {
        // Peak wavelength using sampling temperature
        ScalarFloat lambda_peak = dr::clip(b / m_sampling_temperature,
                                           m_wavelength_range.x() * 1e-9f,
                                           m_wavelength_range.y() * 1e-9f),
                    lambda2_peak = dr::square(lambda_peak),
                    lambda5_peak = dr::square(lambda2_peak) * lambda_peak;

        ScalarFloat P = 1e-9f * c0 / (lambda5_peak *
                    (dr::exp(c1 / (lambda_peak * m_sampling_temperature)) - 1.f));

        ScalarFloat emissivity_max;
        if (m_emissivity_is_constant)
            emissivity_max = dr::slice(m_emissivity_constant);
        else
            emissivity_max = dr::clip(m_emissivity_tex->max(), 0.f, 1.f);

        return P * emissivity_max;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BlackBodySpectrum[" << std::endl
            << "  wavelength_range = [" << m_wavelength_range.x()
            << ", " << m_wavelength_range.y() << "]," << std::endl
            << "  sampling_temperature = " << m_sampling_temperature << "," << std::endl;

        if (m_emissivity_is_constant)
            oss << "  emissivity = " << m_emissivity_constant << "," << std::endl;
        else
            oss << "  emissivity = " << string::indent(m_emissivity_tex) << "," << std::endl;

        if (m_use_attribute) {
            oss << "  temperature_attribute = \"" << m_temperature_attribute << "\"," << std::endl;
        } else {
            oss << "  temperature = " << string::indent(m_temperature_tex) << "," << std::endl;
        }

        oss << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(BlackBodySpectrum)

private:
    void validate_emissivity_spectrum(const Properties::Spectrum &emissivity) const {
        if (emissivity.is_uniform())
            Throw("A blackbody emissivity spectrum must define wavelength-value pairs. "
                  "Use a float for constant emissivity.");

        if (emissivity.wavelengths.size() < 2 ||
            emissivity.wavelengths.size() != emissivity.values.size()) {
            Throw("A blackbody emissivity spectrum must contain at least two "
                  "wavelength-value pairs.");
        }

        for (size_t i = 0; i < emissivity.wavelengths.size(); ++i) {
            if (!std::isfinite(emissivity.wavelengths[i]) ||
                !std::isfinite(emissivity.values[i])) {
                Throw("Blackbody emissivity wavelengths and values must be finite.");
            }
            if (i > 0 && emissivity.wavelengths[i] <= emissivity.wavelengths[i - 1])
                Throw("Blackbody emissivity wavelengths must be strictly increasing.");
        }

        if (emissivity.wavelengths.front() > m_wavelength_range.x() ||
            emissivity.wavelengths.back() < m_wavelength_range.y()) {
            Throw("Blackbody emissivity spectrum range [%f, %f] does not cover "
                  "the configured wavelength range [%f, %f].",
                  emissivity.wavelengths.front(), emissivity.wavelengths.back(),
                  m_wavelength_range.x(), m_wavelength_range.y());
        }
    }

    void validate_emissivity_texture() const {
        if (!m_emissivity_tex)
            Throw("Blackbody emissivity texture is null.");
        if (m_emissivity_tex->is_spatially_varying())
            Throw("Blackbody spectral emissivity must be homogeneous; spatially "
                  "varying emissivity textures are not supported.");

        ScalarVector2f range = m_emissivity_tex->wavelength_range();
        if (!std::isfinite(range.x()) || !std::isfinite(range.y()) ||
            range.x() > m_wavelength_range.x() ||
            range.y() < m_wavelength_range.y()) {
            Throw("Blackbody emissivity spectrum range [%f, %f] does not cover "
                  "the configured wavelength range [%f, %f].",
                  range.x(), range.y(), m_wavelength_range.x(), m_wavelength_range.y());
        }
    }

    // Temperature source
    ref<Texture> m_temperature_tex;       // constant / bitmap path
    std::string  m_temperature_attribute; // vertex attribute name (Kelvins)
    bool         m_use_attribute = false;

    // Homogeneous surface emissivity
    ref<Texture> m_emissivity_tex;
    Float        m_emissivity_constant = 1.f;
    bool         m_emissivity_is_constant = true;

    // Sampling distribution temperature (Kelvins)
    ScalarFloat m_sampling_temperature = ScalarFloat(300.0);

    // Sampling distribution state
    ScalarFloat m_integral_min = 0.f;
    ScalarFloat m_integral = 0.f;
    ScalarVector2f m_wavelength_range;
};

MI_EXPORT_PLUGIN(BlackBodySpectrum)
NAMESPACE_END(mitsuba)
