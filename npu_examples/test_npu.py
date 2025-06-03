#!/usr/bin/env python3
"""
Test OpenVINO NPU Support
This script checks if NPU is available and provides detailed device information
"""

import sys
try:
    import openvino as ov
    print(f"OpenVINO version: {ov.__version__}")
except ImportError:
    print("ERROR: OpenVINO Python module not found!")
    print("Try installing it with: python3 -m pip install openvino")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print("CHECKING OPENVINO DEVICES")
    print("="*50)
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # Get available devices
    devices = core.available_devices
    
    if not devices:
        print("No OpenVINO-compatible devices found!")
        return 1
    
    print(f"Found {len(devices)} device(s):")
    
    # Check for NPU specifically
    npu_found = False
    
    # Check each device in detail
    for device in devices:
        print(f"\n{'-'*20}")
        print(f"Device: {device}")
        
        # Get device properties
        try:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  Full name: {full_name}")
        except:
            pass
            
        try:
            optimization_capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
            print(f"  Optimization capabilities: {optimization_capabilities}")
        except:
            pass

        try:
            available_memory = core.get_property(device, "AVAILABLE_DEVICES")
            print(f"  Available memory: {available_memory}")
        except:
            pass

        # Special check for NPU
        if device.lower() == "npu":
            npu_found = True
            print("  *** NPU DEVICE FOUND ***")
            
    print("\n" + "="*50)
    if not npu_found:
        print("NPU DEVICE NOT FOUND!")
        print("\nPossible reasons:")
        print("1. Your hardware doesn't have an NPU")
        print("2. The NPU driver is not installed or configured correctly")
        print("3. OpenVINO NPU plugin is not installed or configured correctly")
        print("\nTo set up the NPU driver, follow these steps:")
        print("1. Install the NPU driver packages:")
        print("   sudo apt-get install intel-level-zero-npu intel-driver-compiler-npu")
        print("2. Add your user to the render group:")
        print("   sudo usermod -a -G render $USER")
        print("3. Set up udev rules for the NPU device:")
        print("   sudo bash -c \"echo 'SUBSYSTEM==\\\"accel\\\", KERNEL==\\\"accel*\\\", GROUP=\\\"render\\\", MODE=\\\"0660\\\"' > /etc/udev/rules.d/10-intel-vpu.rules\"")
        print("   sudo udevadm control --reload-rules")
        print("   sudo udevadm trigger --subsystem-match=accel")
        print("4. Verify the device permissions:")
        print("   ls -lah /dev/accel/accel0")
        print("5. Reboot your system and try again")
    else:
        print("NPU DEVICE IS AVAILABLE!")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 