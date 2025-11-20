#include <iostream>
#include <openvino/openvino.hpp>

int main() {
    try {
        // Initialize OpenVINO Runtime Core
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
            if (device.find("NPU") != std::string::npos) {
                npu_found = true;
                break;
            }
        }

        if (!npu_found) {
            std::cout << "\nNPU device not found!" << std::endl;
            std::cout << "Possible reasons:" << std::endl;
            std::cout << "1. Your hardware doesn't have an NPU" << std::endl;
            std::cout << "2. The NPU driver is not installed or configured correctly" << std::endl;
            std::cout << "3. OpenVINO NPU plugin is not installed" << std::endl;
        } else {
            std::cout << "\nNPU device is available!" << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
} 