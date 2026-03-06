#include "bitonic_gpu/BitonicSort.hpp"
#include "opencl_core/KernelLoader.hpp"
#include "opencl_core/OpenCLProbe.hpp"
#include "opencl_core/OpenCLProgram.hpp"
#include "opencl_core/OpenCLRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct BenchmarkOptions
{
    std::string kernel_path = "kernels/bitonic_sort.cl";
    int min_exp = 5;
    int max_exp = 20;
    int seeds = 3;
    int warmup = 3;
    int iters = 10;
    int value_min = -1'000'000;
    int value_max = 1'000'000;
    std::string jsonl_out = "artifacts/bench/raw_runs.jsonl";
    bool verify = true;
};

struct RunMetrics
{
    std::size_t size = 0;
    int exponent = 0;
    int seed = 0;
    int iteration = 0;
    bool warmup = false;
    bool correct = false;
    std::uint64_t cpu_sort_ns = 0;
    std::uint64_t gpu_end_to_end_ns = 0;
    std::uint64_t gpu_kernel_ns = 0;
    std::uint64_t gpu_h2d_ns = 0;
    std::uint64_t gpu_d2h_ns = 0;
    std::uint32_t kernel_dispatch_count = 0;
    std::string platform_name;
    std::string device_name;
};

[[noreturn]] void fail_arg(std::string_view message)
{
    throw std::invalid_argument(std::string(message));
}

int parse_int_arg(const char* arg_name, const char* arg_value)
{
    try {
        std::size_t parsed = 0;
        const int value = std::stoi(arg_value, &parsed);
        if (parsed != std::string(arg_value).size()) {
            fail_arg(std::string("Invalid integer for ") + arg_name);
        }
        return value;
    } catch (const std::exception&) {
        fail_arg(std::string("Invalid integer for ") + arg_name);
    }
}

BenchmarkOptions parse_args(int argc, char** argv)
{
    BenchmarkOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fail_arg(std::string("Missing value for ") + name);
            }
            ++i;
            return argv[i];
        };

        if (arg == "--kernel") {
            options.kernel_path = require_value("--kernel");
        } else if (arg == "--min-exp") {
            options.min_exp = parse_int_arg("--min-exp", require_value("--min-exp"));
        } else if (arg == "--max-exp") {
            options.max_exp = parse_int_arg("--max-exp", require_value("--max-exp"));
        } else if (arg == "--seeds") {
            options.seeds = parse_int_arg("--seeds", require_value("--seeds"));
        } else if (arg == "--warmup") {
            options.warmup = parse_int_arg("--warmup", require_value("--warmup"));
        } else if (arg == "--iters") {
            options.iters = parse_int_arg("--iters", require_value("--iters"));
        } else if (arg == "--value-min") {
            options.value_min =
                parse_int_arg("--value-min", require_value("--value-min"));
        } else if (arg == "--value-max") {
            options.value_max =
                parse_int_arg("--value-max", require_value("--value-max"));
        } else if (arg == "--jsonl-out") {
            options.jsonl_out = require_value("--jsonl-out");
        } else if (arg == "--verify") {
            options.verify = true;
        } else if (arg == "--no-verify") {
            options.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bitonic_bench [options]\n"
                << "  --kernel <path>       OpenCL kernel source path\n"
                << "  --min-exp <int>       Minimum exponent for size 2^k\n"
                << "  --max-exp <int>       Maximum exponent for size 2^k\n"
                << "  --seeds <int>         Number of RNG seeds\n"
                << "  --warmup <int>        Warmup runs per (size, seed)\n"
                << "  --iters <int>         Measured runs per (size, seed)\n"
                << "  --value-min <int>     Minimum generated value\n"
                << "  --value-max <int>     Maximum generated value\n"
                << "  --jsonl-out <path>    Output JSONL path\n"
                << "  --verify              Verify GPU output against std::sort\n"
                << "  --no-verify           Disable correctness check\n";
            std::exit(EXIT_SUCCESS);
        } else {
            fail_arg(std::string("Unknown argument: ") + arg);
        }
    }

    if (options.min_exp < 0 || options.max_exp < 0) {
        fail_arg("min-exp and max-exp must be non-negative");
    }
    if (options.min_exp > options.max_exp) {
        fail_arg("min-exp must be <= max-exp");
    }
    if (options.seeds <= 0) {
        fail_arg("seeds must be > 0");
    }
    if (options.warmup < 0) {
        fail_arg("warmup must be >= 0");
    }
    if (options.iters <= 0) {
        fail_arg("iters must be > 0");
    }
    if (options.value_min > options.value_max) {
        fail_arg("value-min must be <= value-max");
    }
    if (options.max_exp >= static_cast<int>(sizeof(std::size_t) * 8)) {
        fail_arg("max-exp is too large for size_t");
    }

    return options;
}

std::size_t size_for_exponent(int exponent)
{
    return static_cast<std::size_t>(1) << exponent;
}

std::vector<int> generate_input(std::size_t size,
                                int seed,
                                int value_min,
                                int value_max)
{
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<int> dist(value_min, value_max);

    std::vector<int> values(size);
    for (int& value : values) {
        value = dist(rng);
    }
    return values;
}

std::string json_escape(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);

    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }

    return escaped;
}

void write_jsonl_record(std::ofstream& out, const RunMetrics& record)
{
    out << "{\"size\":" << record.size << ","
        << "\"exponent\":" << record.exponent << ","
        << "\"seed\":" << record.seed << ","
        << "\"iteration\":" << record.iteration << ","
        << "\"warmup\":" << (record.warmup ? "true" : "false") << ","
        << "\"correct\":" << (record.correct ? "true" : "false") << ","
        << "\"cpu_sort_ns\":" << record.cpu_sort_ns << ","
        << "\"gpu_end_to_end_ns\":" << record.gpu_end_to_end_ns << ","
        << "\"gpu_kernel_ns\":" << record.gpu_kernel_ns << ","
        << "\"gpu_h2d_ns\":" << record.gpu_h2d_ns << ","
        << "\"gpu_d2h_ns\":" << record.gpu_d2h_ns << ","
        << "\"kernel_dispatch_count\":" << record.kernel_dispatch_count << ","
        << "\"platform_name\":\"" << json_escape(record.platform_name) << "\","
        << "\"device_name\":\"" << json_escape(record.device_name) << "\"}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const BenchmarkOptions options = parse_args(argc, argv);

        const bs::OpenCLProbeResult result = bs::probe_opencl();
        if (!result.selection.has_value()) {
            std::cerr << "No suitable OpenCL device found" << std::endl;
            return EXIT_FAILURE;
        }

        const bs::OpenCLRuntime runtime =
            bs::create_opencl_runtime(*result.selection);
        const std::string source = bs::load_text_file(options.kernel_path);
        const cl::Program program = bs::build_program(runtime, source);

        const std::filesystem::path jsonl_path(options.jsonl_out);
        if (!jsonl_path.parent_path().empty()) {
            std::filesystem::create_directories(jsonl_path.parent_path());
        }

        std::ofstream out(jsonl_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "Failed to open output file: " << jsonl_path
                      << std::endl;
            return EXIT_FAILURE;
        }

        for (int exponent = options.min_exp; exponent <= options.max_exp;
             ++exponent) {
            const std::size_t size = size_for_exponent(exponent);
            if (size > std::numeric_limits<cl_uint>::max()) {
                std::cerr << "Size exceeds cl_uint limits: " << size
                          << std::endl;
                return EXIT_FAILURE;
            }

            for (int seed_index = 0; seed_index < options.seeds; ++seed_index) {
                const int seed = seed_index + 1;
                const std::vector<int> input = generate_input(
                    size, seed, options.value_min, options.value_max);

                const int total_runs = options.warmup + options.iters;
                for (int run_index = 0; run_index < total_runs; ++run_index) {
                    const bool warmup = run_index < options.warmup;

                    std::vector<int> cpu_sorted = input;
                    const auto cpu_start = std::chrono::steady_clock::now();
                    std::sort(cpu_sorted.begin(), cpu_sorted.end());
                    const auto cpu_end = std::chrono::steady_clock::now();
                    const std::uint64_t cpu_sort_ns =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                cpu_end - cpu_start)
                                .count());

                    const bs::BitonicRunResult gpu_result =
                        bs::bitonic_sort_opencl_timed(runtime, program, input);

                    const bool correct = (gpu_result.output == cpu_sorted);
                    if (options.verify && !correct) {
                        std::cerr << "Verification failed"
                                  << " size=" << size << " seed=" << seed
                                  << " run=" << (run_index + 1) << std::endl;
                        return EXIT_FAILURE;
                    }

                    const RunMetrics record{
                        .size = size,
                        .exponent = exponent,
                        .seed = seed,
                        .iteration = run_index + 1,
                        .warmup = warmup,
                        .correct = correct,
                        .cpu_sort_ns = cpu_sort_ns,
                        .gpu_end_to_end_ns =
                            gpu_result.timing.gpu_end_to_end_ns,
                        .gpu_kernel_ns = gpu_result.timing.gpu_kernel_ns,
                        .gpu_h2d_ns = gpu_result.timing.gpu_h2d_ns,
                        .gpu_d2h_ns = gpu_result.timing.gpu_d2h_ns,
                        .kernel_dispatch_count =
                            gpu_result.timing.kernel_dispatch_count,
                        .platform_name = result.selection->platform_name,
                        .device_name = result.selection->device_name,
                    };

                    write_jsonl_record(out, record);
                }
            }
        }

        std::cout << "Benchmark completed. JSONL output: " << jsonl_path
                  << std::endl;
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
