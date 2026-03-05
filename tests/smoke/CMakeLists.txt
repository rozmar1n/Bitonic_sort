add_executable(opencl_smoke_test
    OpenCLSmoke.cpp
)

target_include_directories(opencl_smoke_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(opencl_smoke_test
    PRIVATE
        bitonic_gpu
        opencl_core
)

add_test(
    NAME opencl_smoke_test
    COMMAND opencl_smoke_test
)

set_tests_properties(opencl_smoke_test PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
