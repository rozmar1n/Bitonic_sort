#pragma once

#include <bitonic_sort/OpenCLCommon.hpp>
#include <bitonic_sort/OpenCLRuntime.hpp>

#include <vector>

namespace bs {

std::vector<int> run_increment_smoke_test(const OpenCLRuntime& runtime,
                                          const cl::Program& program,
                                          const std::vector<int>& input);

} // namespace bs
