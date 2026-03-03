#include <bitonic_sort/OpenCLProbe.hpp>
#include <iostream>

int main()
{
    try {
        const bs::OpenCLProbeResult result = bs::probe_opencl();

        if (!result.selection.has_value()) {
            std::cerr << "No suitable OpenCL device found" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Selected platform: " << result.selection->platform_name
                  << std::endl;
        std::cout << "Selected device: " << result.selection->device_name
                  << std::endl;
    } catch (const cl::Error& error) {
        std::cerr << "OpenCL error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
