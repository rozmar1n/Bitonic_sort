add_library(bitonic_cpu
    STATIC
    BitonicSort.cpp
)

target_include_directories(bitonic_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(bitonic_cpu
    PUBLIC
        flags
)
