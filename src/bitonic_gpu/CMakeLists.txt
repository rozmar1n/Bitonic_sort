add_library(bitonic_gpu
    STATIC
    BitonicSort.cpp
)

target_include_directories(bitonic_gpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(bitonic_gpu
    PUBLIC
        opencl_core
)
