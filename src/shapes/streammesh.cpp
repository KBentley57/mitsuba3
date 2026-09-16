#include <mitsuba/render/mesh.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/profiler.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class StreamMesh final : public Mesh<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Mesh, m_to_world, from_fields, add_attribute, to_world_scalar)
    MI_IMPORT_TYPES()
    using typename Base::FloatBuffer;
    using typename Base::IndexBuffer;
    using typename Base::InputPoint3f;
    using typename Base::InputNormal3f;

    StreamMesh(const Properties &props) : Base(props) {
        ScopedPhase phase(ProfilerPhase::LoadGeometry);

        // Preserve the AVMC pointer-based input contract. Mesh owns copies of
        // all supplied buffers after this constructor returns.
        auto positions = props.has_property("vertex_positions")
            ? props.get_any<float *>("vertex_positions") : nullptr;
        auto normals = props.has_property("vertex_normals")
            ? props.get_any<float *>("vertex_normals") : nullptr;
        auto uvs = props.has_property("vertex_texcoords")
            ? props.get_any<float *>("vertex_texcoords") : nullptr;
        auto temperatures = props.has_property("vertex_temperature")
            ? props.get_any<float *>("vertex_temperature") : nullptr;
        auto faces = props.has_property("faces")
            ? props.get_any<uint32_t *>("faces") : nullptr;
        int vertex_count = props.get<int>("vertex_count"),
            face_count = props.get<int>("face_count");
        bool pretransformed = props.get<bool>("pretransformed", false),
             trust_normals = props.get<bool>("trust_normals", false),
             flip_uv = props.get<bool>("flip_tex_coords", true);

        if (!positions || vertex_count <= 0)
            Throw("streammesh: positive vertex_count and vertex_positions are required.");
        if (!faces || face_count <= 0)
            Throw("streammesh: positive face_count and faces are required.");

        size_t nv = (size_t) vertex_count, nf = (size_t) face_count;
        ScalarAffineTransform4f to_world = to_world_scalar();
        std::vector<float> position_data(3 * nv), normal_data, uv_data;
        if (normals)
            normal_data.resize(3 * nv);
        if (uvs)
            uv_data.resize(2 * nv);

        for (size_t i = 0; i < nv; ++i) {
            InputPoint3f p = dr::load<InputPoint3f>(positions + 3 * i);
            if (!pretransformed)
                p = to_world * p;
            if (!dr::all(dr::isfinite(p)))
                Throw("streammesh: invalid vertex position at index %zu.", i);
            dr::store(position_data.data() + 3 * i, p);

            if (normals) {
                InputNormal3f n = dr::load<InputNormal3f>(normals + 3 * i);
                if (!(pretransformed && trust_normals))
                    n = dr::normalize(to_world * n);
                if (!dr::all(dr::isfinite(n)) || dr::squared_norm(n) == 0.f)
                    Throw("streammesh: invalid vertex normal at index %zu.", i);
                dr::store(normal_data.data() + 3 * i, n);
            }
            if (uvs) {
                uv_data[2 * i] = uvs[2 * i];
                uv_data[2 * i + 1] = flip_uv ? 1.f - uvs[2 * i + 1] : uvs[2 * i + 1];
            }
        }
        for (size_t i = 0; i < 3 * nf; ++i) {
            if (faces[i] >= (uint32_t) vertex_count)
                Throw("streammesh: face index %u is outside the vertex buffer.", faces[i]);
        }

        TensorXf32 position_tensor(dr::load<FloatBuffer>(position_data.data(), 3 * nv), { nv, 3 });
        TensorXf32 normal_tensor, uv_tensor;
        if (normals)
            normal_tensor = TensorXf32(dr::load<FloatBuffer>(normal_data.data(), 3 * nv), { nv, 3 });
        if (uvs)
            uv_tensor = TensorXf32(dr::load<FloatBuffer>(uv_data.data(), 2 * nv), { nv, 2 });
        TensorXu32 face_tensor(dr::load<IndexBuffer>(faces, 3 * nf), { nf, 3 });

        // from_fields() requires world-space inputs and handles normal/tangent
        // generation, flip_normals, packing, bounds, and initialization.
        m_to_world = new AnimatedTransform4f();
        from_fields(face_tensor, position_tensor, normal_tensor, uv_tensor);
        if (temperatures)
            add_attribute("vertex_temperature",
                          TensorXf32(dr::load<FloatBuffer>(temperatures, nv), { nv, 1 }));
    }

    MI_DECLARE_CLASS(StreamMesh)
};

MI_EXPORT_PLUGIN(StreamMesh)
NAMESPACE_END(mitsuba)
