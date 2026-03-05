#pragma once

#include <opencl_core/detail/OpenCLCommon.hpp>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace bs {

struct OpenCLDeviceInfo
{
    cl::Device handle;
    std::string name;
    std::string vendor;
    cl_device_type type;
};

struct OpenCLPlatformInfo
{
    cl::Platform handle;
    std::string name;
    std::string vendor;
    std::string version;
    std::vector<OpenCLDeviceInfo> devices;
};

struct OpenCLSelection
{
    cl::Platform platform;
    cl::Device device;
    std::string platform_name;
    std::string device_name;
    bool is_gpu;
};

struct OpenCLProbeResult
{
    std::vector<OpenCLPlatformInfo> platforms;
    std::optional<OpenCLSelection> selection = std::nullopt;
};

enum class DevicePreference
{
    GPUOnly,
    GPUThenAny,
    CPUOnly,
    Any
};

OpenCLProbeResult probe_opencl(
    DevicePreference preference = DevicePreference::GPUThenAny);

} // namespace bs
