#include "bitonic_cpu/BitonicSort.hpp"
#include "bitonic_gpu/BitonicSort.hpp"
#include "opencl_core/KernelLoader.hpp"
#include "opencl_core/OpenCLProbe.hpp"
#include "opencl_core/OpenCLProgram.hpp"
#include "opencl_core/OpenCLRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

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
    bool stdin_input = false;
};

struct RunMetrics
{
    std::size_t size = 0;
    int exponent = 0;
    int seed = 0;
    int iteration = 0;
    bool warmup = false;
    bool correct = false;
    bool cpu_bitonic_correct = false;
    bool gpu_correct = false;
    std::uint64_t cpu_sort_ns = 0;
    std::uint64_t cpu_bitonic_ns = 0;
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
        } else if (arg == "--stdin-input") {
            options.stdin_input = true;
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
                << "  --stdin-input         Read input array from stdin for benchmarking\n"
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

bool is_power_of_two(std::size_t value)
{
    return value > 0 && (value & (value - 1U)) == 0;
}

std::size_t next_power_of_two(std::size_t value)
{
    if (value == 0) {
        return 1;
    }

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

int exponent_from_power_of_two_size(std::size_t size)
{
    if (!is_power_of_two(size)) {
        throw std::invalid_argument("size must be a power of two");
    }

    int exponent = 0;
    while (size > 1) {
        size >>= 1U;
        ++exponent;
    }
    return exponent;
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
        << "\"cpu_bitonic_correct\":"
        << (record.cpu_bitonic_correct ? "true" : "false") << ","
        << "\"gpu_correct\":" << (record.gpu_correct ? "true" : "false")
        << ","
        << "\"cpu_sort_ns\":" << record.cpu_sort_ns << ","
        << "\"cpu_bitonic_ns\":" << record.cpu_bitonic_ns << ","
        << "\"gpu_end_to_end_ns\":" << record.gpu_end_to_end_ns << ","
        << "\"gpu_kernel_ns\":" << record.gpu_kernel_ns << ","
        << "\"gpu_h2d_ns\":" << record.gpu_h2d_ns << ","
        << "\"gpu_d2h_ns\":" << record.gpu_d2h_ns << ","
        << "\"kernel_dispatch_count\":" << record.kernel_dispatch_count << ","
        << "\"platform_name\":\"" << json_escape(record.platform_name) << "\","
        << "\"device_name\":\"" << json_escape(record.device_name) << "\"}\n";
}

bool stdin_is_tty()
{
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

bool stdout_is_tty()
{
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

void finish_progress_line_if_needed(bool interactive_progress,
                                    bool& progress_line_active)
{
    if (interactive_progress && progress_line_active) {
        std::cout << std::endl;
        progress_line_active = false;
    }
}

void render_progress_bar(bool interactive_progress,
                         bool& progress_line_active,
                         int completed_runs,
                         int total_runs,
                         double elapsed_s,
                         int exponent,
                         int seed)
{
    if (!interactive_progress || total_runs <= 0) {
        return;
    }

    const double progress_ratio =
        static_cast<double>(completed_runs) / static_cast<double>(total_runs);
    const double progress_pct = progress_ratio * 100.0;
    const int remaining_runs = total_runs - completed_runs;
    const double eta_s =
        completed_runs > 0
            ? elapsed_s * static_cast<double>(remaining_runs) /
                  static_cast<double>(completed_runs)
            : 0.0;

    constexpr int bar_width = 32;
    const int filled_width = std::clamp(
        static_cast<int>(progress_ratio * static_cast<double>(bar_width)),
        0,
        bar_width);

    std::cout << '\r' << "[bench] ["
              << std::string(static_cast<std::size_t>(filled_width), '#')
              << std::string(static_cast<std::size_t>(bar_width - filled_width),
                             '-')
              << "] " << std::fixed << std::setprecision(1)
              << std::setw(5) << progress_pct << "% " << completed_runs << "/"
              << total_runs << " eta=" << eta_s << "s"
              << " size=2^" << exponent << " seed=" << seed << std::flush;

    progress_line_active = true;
    if (completed_runs >= total_runs) {
        std::cout << std::endl;
        progress_line_active = false;
    }
}

void print_default_run_hint_if_needed(int argc, const BenchmarkOptions& options)
{
    if (argc != 1) {
        return;
    }

    std::cout << "[bench] Running default config: min-exp=" << options.min_exp
              << ", max-exp=" << options.max_exp << ", seeds=" << options.seeds
              << ", warmup=" << options.warmup << ", iters=" << options.iters
              << std::endl;
    std::cout << "[bench] This run can take a long time on large sizes."
              << std::endl;

}

void print_stdin_mode_banner(std::size_t original_size,
                             std::size_t working_size,
                             int effective_seeds,
                             int effective_warmup,
                             int effective_iters)
{
    std::cout << "[bench] stdin-input mode: input_size=" << original_size
              << ", working_size=" << working_size << ", seeds="
              << effective_seeds << ", warmup=" << effective_warmup
              << ", iters=" << effective_iters << std::endl;

    if (working_size != original_size) {
        std::cout << "[bench] input was padded to next power-of-two using INT_MAX"
                  << std::endl;
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const BenchmarkOptions options = parse_args(argc, argv);

        const bool auto_stdin_mode = (argc == 1 && !stdin_is_tty());
        bool use_stdin_mode = options.stdin_input || auto_stdin_mode;
        std::vector<int> stdin_original_input;
        std::vector<int> stdin_padded_input;
        std::vector<int> stdin_expected_sorted;

        int effective_seeds = options.seeds;
        int effective_warmup = options.warmup;
        int effective_iters = options.iters;

        if (use_stdin_mode) {
            stdin_original_input = read_ints_from_stdin();
            if (stdin_original_input.empty()) {
                if (options.stdin_input) {
                    std::cerr << "Error: --stdin-input specified, but stdin is empty"
                              << std::endl;
                    return EXIT_FAILURE;
                }

                use_stdin_mode = false;
                std::cerr << "[bench] stdin redirected but empty; falling back to "
                             "synthetic benchmark mode."
                          << std::endl;
            } else {
                stdin_expected_sorted = stdin_original_input;
                std::sort(stdin_expected_sorted.begin(), stdin_expected_sorted.end());
                stdin_padded_input = pad_to_power_of_two(stdin_original_input);

                if (auto_stdin_mode && argc == 1) {
                    effective_seeds = 1;
                    effective_warmup = 0;
                    effective_iters = 1;
                }

                print_stdin_mode_banner(stdin_original_input.size(),
                                        stdin_padded_input.size(),
                                        effective_seeds,
                                        effective_warmup,
                                        effective_iters);
            }
        }

        if (!use_stdin_mode) {
            print_default_run_hint_if_needed(argc, options);
        }

        const int runs_per_seed = effective_warmup + effective_iters;
        const int total_size_points =
            use_stdin_mode ? 1 : (options.max_exp - options.min_exp + 1);
        const int total_runs = total_size_points * effective_seeds * runs_per_seed;
        int completed_runs = 0;
        const auto bench_start = std::chrono::steady_clock::now();
        const bool interactive_progress = stdout_is_tty();
        bool progress_line_active = false;

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

        auto run_for_input = [&](const std::vector<int>& input,
                                 const std::vector<int>& expected_trimmed,
                                 std::size_t original_size,
                                 int exponent,
                                 std::size_t size,
                                 int seed) -> bool {
            const int runs_for_seed = effective_warmup + effective_iters;
            for (int run_index = 0; run_index < runs_for_seed; ++run_index) {
                    const bool warmup = run_index < effective_warmup;

                    std::vector<int> cpu_sorted = input;
                    const auto cpu_start = std::chrono::steady_clock::now();
                    std::sort(cpu_sorted.begin(), cpu_sorted.end());
                    const auto cpu_end = std::chrono::steady_clock::now();
                    const std::uint64_t cpu_sort_ns =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                cpu_end - cpu_start)
                                .count());

                    const auto cpu_bitonic_start =
                        std::chrono::steady_clock::now();
                    const std::vector<int> cpu_bitonic_sorted =
                        bs::bitonic_sort_cpu(input);
                    const auto cpu_bitonic_end = std::chrono::steady_clock::now();
                    const std::uint64_t cpu_bitonic_ns =
                        static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                cpu_bitonic_end - cpu_bitonic_start)
                                .count());

                    const bs::BitonicRunResult gpu_result =
                        bs::bitonic_sort_opencl_timed(runtime, program, input);

                    bool cpu_bitonic_correct = false;
                    bool gpu_correct = false;

                    if (original_size == size) {
                        cpu_bitonic_correct = (cpu_bitonic_sorted == cpu_sorted);
                        gpu_correct = (gpu_result.output == cpu_sorted);
                    } else {
                        std::vector<int> cpu_bitonic_trimmed = cpu_bitonic_sorted;
                        cpu_bitonic_trimmed.resize(original_size);

                        std::vector<int> gpu_trimmed = gpu_result.output;
                        gpu_trimmed.resize(original_size);

                        cpu_bitonic_correct = (cpu_bitonic_trimmed == expected_trimmed);
                        gpu_correct = (gpu_trimmed == expected_trimmed);
                    }

                    const bool correct = cpu_bitonic_correct && gpu_correct;
                    if (options.verify && !correct) {
                        std::cerr << "Verification failed"
                                  << " size=" << size << " seed=" << seed
                                  << " run=" << (run_index + 1)
                                  << " cpu_bitonic_correct="
                                  << (cpu_bitonic_correct ? "true" : "false")
                                  << " gpu_correct="
                                  << (gpu_correct ? "true" : "false")
                                  << std::endl;
                        return false;
                    }

                    const RunMetrics record{
                        .size = size,
                        .exponent = exponent,
                        .seed = seed,
                        .iteration = run_index + 1,
                        .warmup = warmup,
                        .correct = correct,
                        .cpu_bitonic_correct = cpu_bitonic_correct,
                        .gpu_correct = gpu_correct,
                        .cpu_sort_ns = cpu_sort_ns,
                        .cpu_bitonic_ns = cpu_bitonic_ns,
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
                    ++completed_runs;

                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed_s =
                        std::chrono::duration<double>(now - bench_start).count();
                    render_progress_bar(interactive_progress,
                                        progress_line_active,
                                        completed_runs,
                                        total_runs,
                                        elapsed_s,
                                        exponent,
                                        seed);
            }

            return true;
        };

        if (use_stdin_mode) {
            const std::size_t size = stdin_padded_input.size();
            if (size > std::numeric_limits<cl_uint>::max()) {
                std::cerr << "Size exceeds cl_uint limits: " << size << std::endl;
                return EXIT_FAILURE;
            }

            const int exponent = exponent_from_power_of_two_size(size);
            finish_progress_line_if_needed(interactive_progress,
                                           progress_line_active);
            std::cout << "[bench] size=2^" << exponent << " (N=" << size
                      << "): start" << std::endl;
            const auto size_start = std::chrono::steady_clock::now();

            for (int seed_index = 0; seed_index < effective_seeds; ++seed_index) {
                const int seed = seed_index + 1;
                if (!run_for_input(stdin_padded_input,
                                   stdin_expected_sorted,
                                   stdin_original_input.size(),
                                   exponent,
                                   size,
                                   seed)) {
                    return EXIT_FAILURE;
                }

                if (!interactive_progress) {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed_s =
                        std::chrono::duration<double>(now - bench_start).count();
                    const double progress_pct =
                        total_runs > 0
                            ? (100.0 * static_cast<double>(completed_runs) /
                               static_cast<double>(total_runs))
                            : 100.0;
                    std::cout << std::fixed << std::setprecision(1)
                              << "[bench] progress: " << completed_runs << "/"
                              << total_runs << " (" << progress_pct
                              << "%), size=2^" << exponent << ", seed=" << seed
                              << ", elapsed=" << elapsed_s << "s" << std::endl;
                }
            }

            const double size_elapsed_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              size_start)
                    .count();
            finish_progress_line_if_needed(interactive_progress,
                                           progress_line_active);
            std::cout << std::fixed << std::setprecision(2)
                      << "[bench] size=2^" << exponent
                      << ": done in " << size_elapsed_s << "s" << std::endl;
        } else {
            for (int exponent = options.min_exp; exponent <= options.max_exp;
                 ++exponent) {
                const std::size_t size = size_for_exponent(exponent);
                if (size > std::numeric_limits<cl_uint>::max()) {
                    std::cerr << "Size exceeds cl_uint limits: " << size
                              << std::endl;
                    return EXIT_FAILURE;
                }

                finish_progress_line_if_needed(interactive_progress,
                                               progress_line_active);
                std::cout << "[bench] size=2^" << exponent << " (N=" << size
                          << "): start" << std::endl;
                const auto size_start = std::chrono::steady_clock::now();

                for (int seed_index = 0; seed_index < effective_seeds; ++seed_index) {
                    const int seed = seed_index + 1;
                    const std::vector<int> input = generate_input(
                        size, seed, options.value_min, options.value_max);

                    if (!run_for_input(input, {}, size, exponent, size, seed)) {
                        return EXIT_FAILURE;
                    }

                    if (!interactive_progress) {
                        const auto now = std::chrono::steady_clock::now();
                        const double elapsed_s =
                            std::chrono::duration<double>(now - bench_start)
                                .count();
                        const double progress_pct =
                            total_runs > 0
                                ? (100.0 * static_cast<double>(completed_runs) /
                                   static_cast<double>(total_runs))
                                : 100.0;
                        std::cout << std::fixed << std::setprecision(1)
                                  << "[bench] progress: " << completed_runs
                                  << "/" << total_runs << " (" << progress_pct
                                  << "%), size=2^" << exponent
                                  << ", seed=" << seed
                                  << ", elapsed=" << elapsed_s << "s"
                                  << std::endl;
                    }
                }

                const double size_elapsed_s =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - size_start)
                        .count();
                finish_progress_line_if_needed(interactive_progress,
                                               progress_line_active);
                std::cout << std::fixed << std::setprecision(2)
                          << "[bench] size=2^" << exponent
                          << ": done in " << size_elapsed_s << "s" << std::endl;
            }
        }

        finish_progress_line_if_needed(interactive_progress, progress_line_active);
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
