#pragma once

#include "bitonic_sort/OpenCLRuntime.hpp"

#include <string>

namespace bs {

cl::Program build_program(const OpenCLRuntime& runtime,
                          const std::string& source);

} // namespace bs
