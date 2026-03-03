#pragma once

#include <filesystem>
#include <string>

namespace bs {

std::string load_text_file(const std::filesystem::path& path);

} // namespace bs
