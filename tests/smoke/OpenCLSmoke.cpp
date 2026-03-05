#include "bitonic_gpu/BitonicSort.hpp"
#include "opencl_core/KernelLoader.hpp"
#include "opencl_core/OpenCLProbe.hpp"
#include "opencl_core/OpenCLProgram.hpp"
#include "opencl_core/OpenCLRuntime.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
    try {
        const bs::OpenCLProbeResult result = bs::probe_opencl();
        if (!result.selection.has_value()) {
            std::cout << "Smoke skipped: no suitable OpenCL device found"
                      << std::endl;
            return EXIT_SUCCESS;
        }

        const bs::OpenCLRuntime runtime =
            bs::create_opencl_runtime(*result.selection);
        const std::string source =
            bs::load_text_file("kernels/bitonic_sort.cl");
        const cl::Program program = bs::build_program(runtime, source);

        std::vector<int> input = { 7, 1, 9, 3, 2, 6, 5, 4 };
        std::vector<int> expected = input;
        std::sort(expected.begin(), expected.end());

        const std::vector<int> actual =
            bs::bitonic_sort_opencl(runtime, program, input);

        if (actual != expected) {
            std::cerr << "Smoke failed: GPU result differs from std::sort"
                      << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Smoke passed" << std::endl;
        return EXIT_SUCCESS;
    } catch (const cl::Error& error) {
        std::cerr << "OpenCL error: " << error.what() << " (" << error.err()
                  << ")" << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
