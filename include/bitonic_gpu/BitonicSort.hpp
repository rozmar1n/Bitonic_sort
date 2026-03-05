#pragma once

#include <opencl_core/OpenCLRuntime.hpp>
#include <opencl_core/detail/OpenCLCommon.hpp>

#include <vector>

namespace bs {

std::vector<int> bitonic_sort_opencl(const OpenCLRuntime& runtime,
                                     const cl::Program& program,
                                     const std::vector<int>& input);

} // namespace bs
