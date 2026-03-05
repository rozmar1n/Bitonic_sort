#include "bitonic_gpu/BitonicSort.hpp"

#include <limits>
#include <stdexcept>

namespace bs {

std::vector<int> bitonic_sort_opencl(const OpenCLRuntime& runtime,
                                     const cl::Program& program,
                                     const std::vector<int>& input)
{
    if (input.empty()) {
        return {};
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

    cl::Buffer buffer(
        runtime.context, CL_MEM_READ_WRITE, sizeof(int) * output.size());

    runtime.queue.enqueueWriteBuffer(
        buffer, CL_TRUE, 0, sizeof(int) * output.size(), output.data());

    cl::Kernel kernel(program, "bitonic_step");

    for (std::size_t stage = 2; stage <= n; stage <<= 1) {
        kernel.setArg(0, buffer);
        for (std::size_t step = stage >> 1; step > 0; step >>= 1) {
            kernel.setArg(1, static_cast<cl_uint>(stage));
            kernel.setArg(2, static_cast<cl_uint>(step));

            runtime.queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
        }
    }

    runtime.queue.finish();

    runtime.queue.enqueueReadBuffer(
        buffer, CL_TRUE, 0, sizeof(int) * output.size(), output.data());

    return output;
}

} // namespace bs
