set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(mitsuba-bin mitsuba.cpp)

target_link_libraries(mitsuba-bin PRIVATE mitsuba)

target_link_libraries(mitsuba-bin PRIVATE struct-jit)

if (UNIX AND NOT APPLE)
  target_link_libraries(mitsuba-bin PRIVATE dl)
endif()

set_target_properties(mitsuba-bin PROPERTIES OUTPUT_NAME mitsuba)

target_compile_definitions(mitsuba-bin
    PRIVATE
        ADDITIONAL_PLUGIN_PATH=\"${CMAKE_INSTALL_LIBDIR}/mitsuba\")

target_compile_definitions(mitsuba-bin
    PRIVATE
        ADDITIONAL_DATA_PATH=\"${CMAKE_INSTALL_FULL_DATAROOTDIR}/mitsuba\")
