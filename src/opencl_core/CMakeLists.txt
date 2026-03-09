add_library(opencl_core
    STATIC
    OpenCLProbe.cpp
    OpenCLRuntime.cpp
    OpenCLProgram.cpp
    KernelLoader.cpp
)

target_include_directories(opencl_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(opencl_core
    PUBLIC
        flags
        OpenCL::OpenCL
)
