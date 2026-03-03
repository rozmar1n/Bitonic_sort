#pragma once

#include <bitonic_sort/OpenCLCommon.hpp>
#include <bitonic_sort/OpenCLProbe.hpp>

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
