#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/render/mesh.h>
#include <iostream>
#include <stdexcept>
using namespace mitsuba;
using MeshT = Mesh<float, Color<float, 3>>;
using Point3 = Point<float, 3>;
using Vector3 = Vector<float, 3>;
using Transform4 = AffineTransform<Point<float, 4>>;
void check(bool value) { if (!value) throw std::runtime_error("streammesh regression failed"); }
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    Thread::static_initialization();
    Logger::static_initialization();
    Bitmap::static_initialization();
    file_resolver()->prepend(argv[1]);
    for (bool pretransformed : { false, true }) {
        float p[] = {0,0,0, 1,0,0, 0,1,0};
        float n[] = {0,0,1, 0,0,1, 0,0,1};
        float uv[] = {0,0, 1,0, 0,1};
        float temp[] = {300, 330, 360};
        uint32_t f[] = {0,1,2};
        Properties props("streammesh");
        props.set("vertex_count", 3);
        props.set("face_count", 1);
        props.set_any("vertex_positions", (float *)p);
        props.set_any("vertex_normals", (float *)n);
        props.set_any("vertex_texcoords", (float *)uv);
        props.set_any("vertex_temperature", (float *)temp);
        props.set_any("faces", (uint32_t *)f);
        props.set("pretransformed", pretransformed);
        props.set("trust_normals", pretransformed);
        props.set("to_world", Transform4::translate(Vector3(3,4,5)));
        ref<MeshT> mesh = PluginManager::instance()->create_object<MeshT>(props);
        Point3 offset = pretransformed ? Point3(0) : Point3(3,4,5);
        check(mesh->vertex_count() == 3 && mesh->face_count() == 1);
        check(dr::allclose(mesh->vertex_position(0u), offset));
        check(dr::allclose(mesh->vertex_position(1u), offset + Point3(1,0,0)));
        check(dr::allclose(mesh->vertex_normal(0u), Vector3(0,0,1)));
        check(dr::allclose(mesh->vertex_texcoord(0u), Point<float,2>(0,1)));
        SurfaceInteraction<float, Color<float,3>> si = dr::zeros<SurfaceInteraction<float, Color<float,3>>>();
        si.p = offset + Point3(1.f/3.f,1.f/3.f,0);
        si.prim_index = 0;
        check(dr::allclose(mesh->eval_attribute_1("vertex_temperature", si, true), 330.f));
        p[0] = 999; n[2] = -1; uv[0] = 999; temp[0] = 999;
        check(dr::allclose(mesh->vertex_position(0u), offset));
        check(dr::allclose(mesh->eval_attribute_1("vertex_temperature", si, true), 330.f));
    }
    std::cout << "streammesh: transformed and pretransformed buffers, UV flip, normals, temperature interpolation, and ownership passed\n";
}
