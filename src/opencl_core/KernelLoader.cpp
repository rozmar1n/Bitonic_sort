#include "opencl_core/KernelLoader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bs {

std::string load_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

} // namespace bs
