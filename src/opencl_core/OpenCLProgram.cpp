#include "opencl_core/OpenCLProgram.hpp"

namespace bs {
cl::Program build_program(const OpenCLRuntime& runtime,
                          const std::string& source)
{
    cl::Program::Sources sources{ { source.data(), source.size() } };

    cl::Program program(runtime.context, sources);

    try {
        program.build({ runtime.device });
    } catch (const cl::Error& error) {
        const std::string build_log =
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(runtime.device);

        throw std::runtime_error("Failed to build OpenCL program:\n" +
                                 build_log);
    }

    return program;
}
} // namespace bs
