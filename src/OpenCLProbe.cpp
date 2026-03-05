#include <opencl_core/OpenCLProbe.hpp>

namespace bs {

namespace {

OpenCLSelection make_selection(const OpenCLPlatformInfo& platform_info,
                               const OpenCLDeviceInfo& device_info)
{
    const bool is_gpu = (device_info.type & CL_DEVICE_TYPE_GPU) != 0;

    return OpenCLSelection{
        .platform = platform_info.handle,
        .device = device_info.handle,
        .platform_name = platform_info.name,
        .device_name = device_info.name,
        .is_gpu = is_gpu,
    };
}

} // namespace

OpenCLProbeResult probe_opencl(DevicePreference preference)
{
    OpenCLProbeResult result;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    result.platforms.reserve(platforms.size());

    std::optional<OpenCLSelection> first_gpu = std::nullopt;
    std::optional<OpenCLSelection> first_cpu = std::nullopt;
    std::optional<OpenCLSelection> first_any = std::nullopt;

    for (const auto& platform : platforms) {
        OpenCLPlatformInfo res_platform;
        res_platform.handle = platform;
        res_platform.name = platform.getInfo<CL_PLATFORM_NAME>();
        res_platform.vendor = platform.getInfo<CL_PLATFORM_VENDOR>();
        res_platform.version = platform.getInfo<CL_PLATFORM_VERSION>();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) {
            continue;
        }

        for (const auto& device : devices) {
            OpenCLDeviceInfo res_device;
            res_device.handle = device;
            res_device.name = device.getInfo<CL_DEVICE_NAME>();
            res_device.vendor = device.getInfo<CL_DEVICE_VENDOR>();
            res_device.type = device.getInfo<CL_DEVICE_TYPE>();

            const bool is_gpu = (res_device.type & CL_DEVICE_TYPE_GPU) != 0;
            const bool is_cpu = (res_device.type & CL_DEVICE_TYPE_CPU) != 0;

            OpenCLSelection selection_candidate =
                make_selection(res_platform, res_device);

            if (!first_any.has_value()) {
                first_any = selection_candidate;
            }

            if (is_gpu && !first_gpu.has_value()) {
                first_gpu = selection_candidate;
            }

            if (is_cpu && !first_cpu.has_value()) {
                first_cpu = selection_candidate;
            }

            res_platform.devices.emplace_back(std::move(res_device));
        }

        result.platforms.emplace_back(std::move(res_platform));
    }

    switch (preference) {
        case DevicePreference::GPUOnly:
            result.selection = first_gpu;
            break;
        case DevicePreference::GPUThenAny:
            result.selection = first_gpu.has_value() ? first_gpu : first_any;
            break;
        case DevicePreference::CPUOnly:
            result.selection = first_cpu;
            break;
        case DevicePreference::Any:
            result.selection = first_any;
            break;
    }

    return result;
}
} // namespace bs
