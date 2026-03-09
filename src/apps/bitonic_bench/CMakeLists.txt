add_executable(bitonic_bench
    main.cpp
)

target_include_directories(bitonic_bench
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(bitonic_bench
    PRIVATE
        bitonic_cpu
        bitonic_gpu
        opencl_core
)
