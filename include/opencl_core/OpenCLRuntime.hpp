#pragma once

#include <opencl_core/OpenCLProbe.hpp>
#include <opencl_core/detail/OpenCLCommon.hpp>

namespace bs {

struct OpenCLRuntime
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
};

OpenCLRuntime create_opencl_runtime(const OpenCLSelection& selection);

} // namespace bs
