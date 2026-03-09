add_executable(bitonic_sort_cli
    main.cpp
)

target_include_directories(bitonic_sort_cli
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(bitonic_sort_cli
    PRIVATE
        bitonic_gpu
        opencl_core
)
