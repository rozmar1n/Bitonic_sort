#include "bitonic_gpu/BitonicSort.hpp"
#include "opencl_core/KernelLoader.hpp"
#include "opencl_core/OpenCLProbe.hpp"
#include "opencl_core/OpenCLProgram.hpp"
#include "opencl_core/OpenCLRuntime.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool is_power_of_two(std::size_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

std::size_t next_power_of_two(std::size_t value)
{
    std::size_t result = 1;
    while (result < value) {
        if (result > (std::numeric_limits<std::size_t>::max() / 2U)) {
            throw std::overflow_error("Input size is too large");
        }
        result *= 2U;
    }
    return result;
}

std::vector<int> read_ints_from_stdin()
{
    std::vector<int> values;
    int value = 0;
    while (std::cin >> value) {
        values.push_back(value);
    }

    if (!std::cin.eof()) {
        throw std::runtime_error("Failed to parse integer from stdin");
    }

    return values;
}

std::vector<int> pad_to_power_of_two(const std::vector<int>& input)
{
    if (input.empty() || is_power_of_two(input.size())) {
        return input;
    }

    const std::size_t padded_size = next_power_of_two(input.size());
    std::vector<int> padded = input;
    padded.resize(padded_size, std::numeric_limits<int>::max());
    return padded;
}

void print_sorted_stdout(const std::vector<int>& values)
{
    if (values.empty()) {
        std::cout << '\n';
        return;
    }

    std::cout << values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        std::cout << ' ' << values[i];
    }
    std::cout << '\n';
}

} // namespace

int main()
{
    try {
        const std::vector<int> input = read_ints_from_stdin();
        if (input.empty()) {
            std::cerr << "Input is empty" << std::endl;
            return EXIT_FAILURE;
        }

        const std::size_t original_size = input.size();
        const std::vector<int> padded_input = pad_to_power_of_two(input);

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

        std::vector<int> gpu_sorted =
            bitonic_sort_opencl(runtime, program, padded_input);
        gpu_sorted.resize(original_size);
        print_sorted_stdout(gpu_sorted);
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
