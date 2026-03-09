#include "bitonic_cpu/BitonicSort.hpp"

#include <stdexcept>
#include <utility>

namespace bs {

namespace {

bool is_power_of_two(std::size_t value)
{
    return value > 0 && (value & (value - 1U)) == 0;
}

} // namespace

std::vector<int> bitonic_sort_cpu(const std::vector<int>& input)
{
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Input size must be a power of two");
    }

    std::vector<int> output = input;

    for (std::size_t stage = 2; stage <= n; stage <<= 1U) {
        for (std::size_t step = stage >> 1U; step > 0; step >>= 1U) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t j = i ^ step;
                if (j <= i) {
                    continue;
                }

                const bool ascending = (i & stage) == 0;
                if ((ascending && output[i] > output[j]) ||
                    (!ascending && output[i] < output[j])) {
                    std::swap(output[i], output[j]);
                }
            }
        }
    }

    return output;
}

} // namespace bs
