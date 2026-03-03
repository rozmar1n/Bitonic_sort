#include "bitonic_sort/OpenCLRuntime.hpp"

namespace bs {

OpenCLRuntime create_opencl_runtime(const OpenCLSelection& selection)
{
    cl::Context context(selection.device);

    cl::CommandQueue queue(
        context, selection.device, CL_QUEUE_PROFILING_ENABLE);

    return OpenCLRuntime{
        .platform = selection.platform,
        .device = selection.device,
        .context = context,
        .queue = queue,
    };
}

} // namespace bs
