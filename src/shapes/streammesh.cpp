
#include <mitsuba/render/mesh.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/profiler.h>

#include <array>

NAMESPACE_BEGIN(mitsuba)

template <class Float, class Spectrum>
class StreamMesh final : public Mesh<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Mesh, m_name, m_bbox, m_to_world, m_vertex_count,
                   m_face_count, m_vertex_positions, m_vertex_normals,
                   m_vertex_texcoords, m_faces, add_attribute, m_face_normals,
                   recompute_vertex_normals, has_vertex_normals, initialize)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;
    using typename Base::InputFloat;
    using typename Base::InputPoint3f;
    using typename Base::InputVector2f;
    using typename Base::InputVector3f;
    using typename Base::InputNormal3f;
    using typename Base::FloatStorage;

    StreamMesh( const mitsuba::Properties& props ) : Base(props) {
        float* positions = props.has_property("vertex_positions")
            ? props.get_any<float*>("vertex_positions") : nullptr;
        float* normals = props.has_property("vertex_normals")
            ? props.get_any<float*>("vertex_normals") : nullptr;
        float* uvs = props.has_property("vertex_texcoords")
            ? props.get_any<float*>("vertex_texcoords") : nullptr;
        float* temps = props.has_property("vertex_temperature")
            ? props.get_any<float*>("vertex_temperature") : nullptr;
        std::uint32_t* faces = props.has_property("faces")
            ? props.get_any<std::uint32_t*>("faces") : nullptr;
        int vertex_count = props.get<int>("vertex_count");
        int face_count = props.get<int>("face_count");
        bool pretransformed = props.get<bool>("pretransformed", false);
        bool trust_normals = props.get<bool>("trust_normals", false);

        init( positions
            , normals
            , uvs
            , temps
            , faces
            , vertex_count
            , face_count
            , props.get<bool>("flip_tex_coords", true)
            , pretransformed
            , trust_normals );
    }

    void init( float* in_positions
             , float* in_normals
             , float* in_uvs
             , float* in_temps
             , std::uint32_t* in_faces
             , std::uint32_t vertex_count
             , std::uint32_t face_count
             , bool flip_tex_coords=false
             , bool pretransformed=false
             , bool trust_normals=false )
    {
      ScopedPhase phase(ProfilerPhase::LoadGeometry);

      using ScalarIndex3 = std::array<ScalarIndex, 3>;

      if (!in_positions || vertex_count <= 0) {
        Throw("no vertex positions!");
      }
      if (!in_faces || face_count <= 0) {
        Throw("no face indices!");
      }

      m_vertex_count = (ScalarIndex)vertex_count;
      m_face_count = (ScalarSize)face_count;

      if (pretransformed) {
        for (std::size_t i = 0; i < vertex_count; i++)
        {
          InputPoint3f p{ in_positions[3*i + 0]
                        , in_positions[3*i + 1]
                        , in_positions[3*i + 2] };
          if (unlikely(!all(dr::isfinite(p)))) {
              Throw("mesh contains invalid vertex position data");
          }
          m_bbox.expand(p);
        }
        m_vertex_positions = dr::load<FloatStorage>(in_positions, m_vertex_count * 3);
      } else {
        std::unique_ptr<float[]> vertices(new float[vertex_count * 3]);
        for (std::size_t i = 0; i < vertex_count; i++)
        {
          InputPoint3f p{ in_positions[3*i + 0]
                        , in_positions[3*i + 1]
                        , in_positions[3*i + 2] };
          p = m_to_world.scalar() * p;
          if (unlikely(!all(dr::isfinite(p)))) {
              Throw("mesh contains invalid vertex position data");
          }
          m_bbox.expand(p);
          InputVector3f pv = p;
          InputFloat* ptr = vertices.get() + (3*i);
          dr::store(ptr, pv);
        }
        m_vertex_positions = dr::load<FloatStorage>(vertices.get(), m_vertex_count * 3);
      }

      if (in_normals)
      {
        if (pretransformed && trust_normals) {
          if (!m_face_normals) {
              m_vertex_normals = dr::load<FloatStorage>(in_normals, m_vertex_count * 3);
          }
        } else {
          std::unique_ptr<float[]> normals(new float[vertex_count * 3]);
          for (std::size_t i = 0; i < vertex_count; i++)
          {
            InputNormal3f n{ in_normals[3*i + 0]
                           , in_normals[3*i + 1]
                           , in_normals[3*i + 2] };
            n = dr::normalize(m_to_world.scalar() * n);
            if (unlikely(!all(dr::isfinite(n)))) {
                Throw("mesh contains invalid vertex normal data");
            }
            InputFloat* ptr = normals.get() + (3*i);
            dr::store(ptr, n);
          }
          if (!m_face_normals) {
              m_vertex_normals = dr::load<FloatStorage>(normals.get(), m_vertex_count * 3);
          }
        }
      }

      if (in_uvs)
      {
        std::unique_ptr<float[]> texcoords(new float[vertex_count * 2]);
        for (std::size_t i = 0; i < vertex_count; i++)
        {
          InputVector2f uv{ in_uvs[2*i + 0]
                          , in_uvs[2*i + 1] };
          if (flip_tex_coords)
              uv.y() = 1.f - uv.y();
          InputFloat* ptr = texcoords.get() + (2*i);
          dr::store(ptr, uv);
        }
        m_vertex_texcoords = dr::load<FloatStorage>(texcoords.get(), m_vertex_count * 2);
      }

      if (in_temps)
      {
        std::vector<InputFloat> temps(vertex_count);
        for (std::size_t i = 0; i < vertex_count; i++) {
          temps[i] = in_temps[i];
        }
        add_attribute("vertex_temperature", 1, temps);
      }

      m_faces = dr::load<DynamicBuffer<UInt32>>(in_faces, m_face_count * 3);

      if (!m_face_normals && !in_normals) {
          recompute_vertex_normals();
      }

      initialize();
    }

    MI_DECLARE_CLASS(StreamMesh)
};

MI_EXPORT_PLUGIN(StreamMesh)

NAMESPACE_END(mitsuba)
