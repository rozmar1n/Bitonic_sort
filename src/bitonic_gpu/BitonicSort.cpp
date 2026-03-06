#include "bitonic_gpu/BitonicSort.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bs {

namespace {

std::uint64_t profiling_duration_ns(const cl::Event& event)
{
    const cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return static_cast<std::uint64_t>(end - start);
}

} // namespace

BitonicRunResult bitonic_sort_opencl_timed(const OpenCLRuntime& runtime,
                                           const cl::Program& program,
                                           const std::vector<int>& input)
{
    if (input.empty()) {
        return BitonicRunResult{};
    }

    const std::size_t n = input.size();
    const bool is_power_of_two = (n & (n - 1)) == 0;

    if (!is_power_of_two) {
        throw std::invalid_argument("Input size must be a power of two");
    }

    if (n > std::numeric_limits<cl_uint>::max()) {
        throw std::invalid_argument("Input size too large for cl_uint");
    }

    std::vector<int> output = input;

    const auto start = std::chrono::steady_clock::now();

    cl::Buffer buffer(
        runtime.context, CL_MEM_READ_WRITE, sizeof(int) * output.size());

    cl::Event write_event;
    runtime.queue.enqueueWriteBuffer(buffer,
                                     CL_TRUE,
                                     0,
                                     sizeof(int) * output.size(),
                                     output.data(),
                                     nullptr,
                                     &write_event);

    cl::Kernel kernel(program, "bitonic_step");
    std::vector<cl::Event> kernel_events;
    std::uint32_t kernel_dispatch_count = 0;

    for (std::size_t stage = 2; stage <= n; stage <<= 1) {
        kernel.setArg(0, buffer);
        for (std::size_t step = stage >> 1; step > 0; step >>= 1) {
            kernel.setArg(1, static_cast<cl_uint>(stage));
            kernel.setArg(2, static_cast<cl_uint>(step));

            cl::Event kernel_event;
            runtime.queue.enqueueNDRangeKernel(kernel,
                                               cl::NullRange,
                                               cl::NDRange(n),
                                               cl::NullRange,
                                               nullptr,
                                               &kernel_event);
            kernel_events.push_back(std::move(kernel_event));
            ++kernel_dispatch_count;
        }
    }

    runtime.queue.finish();

    cl::Event read_event;
    runtime.queue.enqueueReadBuffer(buffer,
                                    CL_TRUE,
                                    0,
                                    sizeof(int) * output.size(),
                                    output.data(),
                                    nullptr,
                                    &read_event);

    const auto end = std::chrono::steady_clock::now();

    std::uint64_t kernel_ns = 0;
    for (const cl::Event& event : kernel_events) {
        kernel_ns += profiling_duration_ns(event);
    }

    const std::uint64_t h2d_ns = profiling_duration_ns(write_event);
    const std::uint64_t d2h_ns = profiling_duration_ns(read_event);
    const std::uint64_t gpu_end_to_end_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());

    return BitonicRunResult{
        .output = std::move(output),
        .timing =
            BitonicTimingMetrics{
                .gpu_end_to_end_ns = gpu_end_to_end_ns,
                .gpu_kernel_ns = kernel_ns,
                .gpu_h2d_ns = h2d_ns,
                .gpu_d2h_ns = d2h_ns,
                .kernel_dispatch_count = kernel_dispatch_count,
            },
    };
}

std::vector<int> bitonic_sort_opencl(const OpenCLRuntime& runtime,
                                     const cl::Program& program,
                                     const std::vector<int>& input)
{
    return bitonic_sort_opencl_timed(runtime, program, input).output;
}

} // namespace bs
