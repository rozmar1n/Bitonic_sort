#pragma once

#include <opencl_core/OpenCLRuntime.hpp>
#include <opencl_core/detail/OpenCLCommon.hpp>

#include <vector>

namespace bs {

std::vector<int> run_increment_smoke_test(const OpenCLRuntime& runtime,
                                          const cl::Program& program,
                                          const std::vector<int>& input);

} // namespace bs
