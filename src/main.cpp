#include "bitonic_gpu/BitonicSort.hpp"
#include "opencl_core/KernelLoader.hpp"
#include "opencl_core/OpenCLProbe.hpp"
#include "opencl_core/OpenCLProgram.hpp"
#include "opencl_core/OpenCLRuntime.hpp"

#include <algorithm>
#include <cstddef>
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

        std::vector<int> input = { 5, 4, 3, 2, 1, 2,  3,  4,
                                   5, 6, 7, 8, 9, 10, 11, 883 };

        std::vector<int> cpu_input(input);
        auto gpu_sorted = bitonic_sort_opencl(runtime, program, input);
        std::sort(cpu_input.begin(), cpu_input.end());

        for (std::size_t i = 0; i < input.size(); ++i) {
            if (gpu_sorted[i] != cpu_input[i]) {
                std::cerr << "Mismatch at index " << i << std::endl;
                return EXIT_FAILURE;
            }
        }
        std::cout << "Bitonic sort finished successfully" << std::endl;
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
