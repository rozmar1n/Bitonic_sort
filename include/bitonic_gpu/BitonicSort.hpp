#pragma once

#include <opencl_core/OpenCLRuntime.hpp>
#include <opencl_core/detail/OpenCLCommon.hpp>

#include <cstdint>
#include <vector>

namespace bs {

struct BitonicTimingMetrics
{
    std::uint64_t gpu_end_to_end_ns = 0;
    std::uint64_t gpu_kernel_ns = 0;
    std::uint64_t gpu_h2d_ns = 0;
    std::uint64_t gpu_d2h_ns = 0;
    std::uint32_t kernel_dispatch_count = 0;
};

struct BitonicRunResult
{
    std::vector<int> output;
    BitonicTimingMetrics timing;
};

BitonicRunResult bitonic_sort_opencl_timed(const OpenCLRuntime& runtime,
                                           const cl::Program& program,
                                           const std::vector<int>& input);

std::vector<int> bitonic_sort_opencl(const OpenCLRuntime& runtime,
                                     const cl::Program& program,
                                     const std::vector<int>& input);

} // namespace bs
