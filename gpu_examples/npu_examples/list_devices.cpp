#include <iostream>
#include <openvino/openvino.hpp>

int main() {
    try {
        // Initialize OpenVINO runtime
        ov::Core core;

        // Get available devices
        std::cout << "Available OpenVINO devices:" << std::endl;
        for (const auto& device : core.get_available_devices()) {
            std::cout << " - " << device;
            
            // Try to get device full name
            try {
                std::string fullName = core.get_property(device, ov::device::full_name);
                std::cout << " (" << fullName << ")";
            } catch (...) {
                // Skip if property not available
            }
            
            std::cout << std::endl;
        }

        // Check specifically for NPU
        bool npu_found = false;
        for (const auto& device : core.get_available_devices()) {
            if (device == "NPU") {
                npu_found = true;
                break;
            }
        }

        if (npu_found) {
            std::cout << "\nNPU device is available!" << std::endl;
        } else {
            std::cout << "\nNPU device is NOT available. Possible reasons:" << std::endl;
            std::cout << "1. Your hardware doesn't have an Intel NPU" << std::endl;
            std::cout << "2. NPU drivers are not installed correctly" << std::endl;
            std::cout << "3. OpenVINO NPU plugin is not properly configured" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 