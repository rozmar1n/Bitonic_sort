#include "bitonic_sort/OpenCLSmoke.hpp"

namespace bs {

std::vector<int> run_increment_smoke_test(const OpenCLRuntime& runtime,
                                          const cl::Program& program,
                                          const std::vector<int>& input)
{
    if (input.empty()) {
        return {};
    }

    std::vector<int> output = input;

    cl::Buffer buffer(
        runtime.context, CL_MEM_READ_WRITE, sizeof(int) * output.size());

    runtime.queue.enqueueWriteBuffer(
        buffer, CL_TRUE, 0, sizeof(int) * output.size(), output.data());

    cl::Kernel kernel(program, "increment_kernel");
    kernel.setArg(0, buffer);

    runtime.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange);

    runtime.queue.finish();

    runtime.queue.enqueueReadBuffer(
        buffer, CL_TRUE, 0, sizeof(int) * output.size(), output.data());

    return output;
}

} // namespace bs
