# ----------------------------------------------------------
#  Mitsuba library
# ----------------------------------------------------------

# Some parts of Mitsuba use the TSL hash map class
include_directories(../ext/nanobind/ext/robin_map/include)

add_subdirectory(core)
add_subdirectory(render)

add_library(mitsuba SHARED)
target_link_libraries(mitsuba PRIVATE mitsuba-core mitsuba-render)

# Re-export the PUBLIC deps from the OBJECT libs onto `mitsuba` itself so
# they appear in MitsubaTargets.cmake (without the OBJECT libs themselves).
target_link_libraries(mitsuba PUBLIC
  nanothread
  JPEG::JPEG
  PNG::PNG
  TIFF::TIFF
  drjit
  fast_float)

if(MI_EXTERNAL_OPENEXR)
  target_link_libraries(mitsuba PUBLIC ${MI_OPENEXR_TARGET})
else()
  target_link_libraries(mitsuba PUBLIC IlmImf)
endif()

if(MI_EXTERNAL_PUGIXML)
  target_link_libraries(mitsuba PUBLIC pugixml::pugixml)
else()
  target_link_libraries(mitsuba PUBLIC pugixml)
endif()

if (MI_ENABLE_JIT)
  target_link_libraries(mitsuba PUBLIC drjit-core)
endif()

if (MI_ENABLE_AUTODIFF)
  target_link_libraries(mitsuba PUBLIC drjit-extra)
endif()

set_target_properties(mitsuba PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU)$")
  # Treat undefined symbols as errors when linking
  target_link_libraries(mitsuba PRIVATE -Wl,--no-undefined)
  # Silence an incorrect warning message in GCC LTO builds
  target_link_options(mitsuba PRIVATE -Walloc-size-larger-than=18446744073709551615)
endif()

# ----------------------------------------------------------
#  Mitsuba executable
# ----------------------------------------------------------

add_subdirectory(mitsuba)

# ----------------------------------------------------------
#  Plugins
# ----------------------------------------------------------

add_subdirectory(bsdfs)
add_subdirectory(emitters)
add_subdirectory(films)
add_subdirectory(integrators)
add_subdirectory(media)
add_subdirectory(phase)
add_subdirectory(rfilters)
add_subdirectory(samplers)
add_subdirectory(sensors)
add_subdirectory(shapes)
add_subdirectory(spectra)
add_subdirectory(textures)
add_subdirectory(volumes)
set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)

# ----------------------------------------------------------
#  Python bindings and extensions
# ----------------------------------------------------------

if (MI_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
