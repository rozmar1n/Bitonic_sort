#include "bitonic_sort/KernelLoader.hpp"
#include "bitonic_sort/OpenCLProgram.hpp"
#include <bitonic_sort/OpenCLProbe.hpp>
#include <bitonic_sort/OpenCLRuntime.hpp>

#include <iostream>

int main()
{
    try {
        const bs::OpenCLProbeResult result = bs::probe_opencl();

        if (!result.selection.has_value()) {
            std::cerr << "No suitable OpenCL device found" << std::endl;
            return EXIT_FAILURE;
        }

        const bs::OpenCLRuntime runtime =
            bs::create_opencl_runtime(*result.selection);

        const std::string source =
            bs::load_text_file("kernels/bitonic_sort.cl");

        const cl::Program program = bs::build_program(runtime, source);

        (void)program;

        std::cout << "Selected platform: " << result.selection->platform_name
                  << std::endl;
        std::cout << "Selected device: " << result.selection->device_name
                  << std::endl;
        std::cout << "OpenCL context and command created successfully"
                  << std::endl;
        std::cout << "OpenCL program built successfully" << std::endl;

        (void)runtime;
    } catch (const cl::Error& error) {
        std::cerr << "OpenCL error: " << error.what() << " (" << error.err()
                  << ")" << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
