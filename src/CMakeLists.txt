add_executable(bitonic_sort_cli
    main.cpp
)

target_link_libraries(bitonic_sort_cli
    flags
    OpenCL::OpenCL
)
