#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static const char* device_type_to_string(cl_device_type type)
{
    if (type & CL_DEVICE_TYPE_GPU) {
        return "GPU";
    }
    if (type & CL_DEVICE_TYPE_CPU) {
        return "CPU";
    }
    if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        return "ACCELERATOR";
    }

    return "OTHER";
}

int main()
{
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Found" << platforms.size() << "OpenCL plarform(s)"
                  << std::endl;

        for (const auto& platform : platforms) {
            const std::string platform_name =
                platform.getInfo<CL_PLATFORM_NAME>();
            const std::string plarform_vendor =
                platform.getInfo<CL_PLATFORM_VENDOR>();
            const std::string platform_version =
                platform.getInfo<CL_PLATFORM_VERSION>();

            std::cout << "\nPlarform: " << platform_name << std::endl;
            std::cout << "Vendor: " << plarform_vendor << std::endl;
            std::cout << "Version: " << platform_version << std::endl;

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            if (devices.empty()) {
                platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            }

            if (devices.empty()) {
                std::cerr << "No OpenCL devices found" << std::endl;
                continue;
            }

            for (const auto& device : devices) {
                const std::string device_name =
                    device.getInfo<CL_DEVICE_NAME>();
                const std::string device_vendor =
                    device.getInfo<CL_DEVICE_VENDOR>();
                const cl_device_type device_type =
                    device.getInfo<CL_DEVICE_TYPE>();

                std::cout << " Device: " << device_name << std::endl;
                std::cout << " Vendor: " << device_vendor << std::endl;
                std::cout << " Type: " << device_type_to_string(device_type)
                          << std::endl;
            }
        }
    } catch (const cl::Error& error) {
        std::cerr << "OpenCL error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
