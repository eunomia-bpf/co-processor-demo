#!/usr/bin/env python3
"""
OpenVINO Test Script - Checks installation and available devices
"""

import sys
import openvino as ov
import numpy as np

def main():
    print("OpenVINO Version:", ov.__version__)
    print("\n" + "="*50)
    print("CHECKING AVAILABLE DEVICES")
    print("="*50)
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # Get available devices
    devices = core.available_devices
    
    if not devices:
        print("No OpenVINO-compatible devices found!")
        return 1
    
    print(f"Found {len(devices)} device(s):")
    
    # Check each device in detail
    for device in devices:
        print(f"\n{'-'*20}")
        print(f"Device: {device}")
        
        # Get device properties
        full_device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"Full name: {full_device_name}")
        
        try:
            optimization_capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
            print(f"Optimization capabilities: {optimization_capabilities}")
        except:
            print("Optimization capabilities: Not available")
        
        try:
            supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
            if supported_properties:
                print("Supported properties:")
                for prop in sorted(supported_properties):
                    try:
                        value = core.get_property(device, prop)
                        print(f"  - {prop}: {value}")
                    except:
                        print(f"  - {prop}: <unable to get value>")
        except:
            print("Supported properties: Not available")
    
    # Specifically check for NPU
    if "NPU" in devices:
        print("\n" + "="*50)
        print("NPU DEVICE FOUND!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("NPU DEVICE NOT FOUND")
        print("Possible reasons:")
        print("1. Your hardware doesn't have an Intel NPU")
        print("2. NPU drivers are not installed or not working properly")
        print("3. OpenVINO NPU plugin is not installed correctly")
        print("="*50)
    
    # Test a simple inference task
    print("\nRunning simple inference test on CPU...")
    
    # Create a simple model (add operation)
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    # Create the model
    input1 = ov.opset13.parameter([2, 2], dtype=ov.Type.f32)
    input2 = ov.opset13.parameter([2, 2], dtype=ov.Type.f32)
    add = ov.opset13.add(input1, input2)
    model = ov.Model([add], [input1, input2], "addition")
    
    # Compile for CPU
    compiled_model = core.compile_model(model, "CPU")
    
    # Create inference request
    infer_request = compiled_model.create_infer_request()
    
    # Set inputs
    infer_request.inputs = {0: a, 1: b}
    
    # Perform inference
    infer_request.infer()
    
    # Get result
    result = infer_request.outputs[0]
    
    print("Input A:")
    print(a)
    print("Input B:")
    print(b)
    print("Result (A + B):")
    print(result)
    
    # Verify result
    expected = a + b
    if np.array_equal(result, expected):
        print("Test PASSED!")
    else:
        print("Test FAILED!")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 