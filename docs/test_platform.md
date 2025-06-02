# Test Platform Specifications

## System Overview

This OpenCL co-processor demo was tested on a system featuring the Intel® Core™ Ultra 7 Processor 258V, which includes both an integrated Intel® Arc™ GPU and a dedicated Neural Processing Unit (NPU) that serve as co-processors.

## Processor Specifications

- **Processor Model**: Intel® Core™ Ultra 7 258V
- **Code Name**: Products formerly Lunar Lake
- **Launch Date**: Q3'24
- **Lithography**: TSMC N3B
- **Overall Peak TOPS (Int8)**: 115

### CPU Specifications
- **Total Cores**: 8 (4 Performance + 4 Efficient cores)
- **Total Threads**: 8
- **Max Turbo Frequency**: 4.8 GHz
- **Cache**: 12 MB Intel® Smart Cache
- **Base Power**: 17W
- **Maximum Turbo Power**: 37W

### GPU Co-Processor (Intel® Arc™ 140V)
- **GPU Name**: Intel® Arc™ 140V GPU
- **Graphics Max Dynamic Frequency**: 1.95 GHz
- **GPU Peak TOPS (Int8)**: 64
- **Xe-cores**: 8
- **OpenCL Support**: 3.0
- **Ray Tracing**: Yes
- **Device ID**: 0x64A0

### NPU Co-Processor (Intel® AI Boost)
- **NPU Name**: Intel® AI Boost
- **NPU Peak TOPS (Int8)**: 47
- **Sparsity Support**: Yes
- **Windows Studio Effects Support**: Yes
- **AI Software Framework Support**:
  - OpenVINO™
  - WindowsML
  - DirectML
  - ONNX RT
  - WebNN

### Memory Specifications
- **Max Memory Size**: 32 GB
- **Memory Types**: LPDDR5X up to 8533 MT/s
- **Memory Channels**: 2

## Co-Processor Capabilities

### GPU Computing Features
- **OpenCL Version**: 3.0
- **AI Acceleration**: Yes (Intel® Deep Learning Boost)
- **AI Software Framework Support**: 
  - OpenVINO™
  - WindowsML
  - DirectML
  - ONNX RT
  - WebGPU
  - WebNN

### NPU Computing Features
The Neural Processing Unit (NPU) is specifically designed for AI and machine learning workloads:
- Dedicated AI acceleration with 47 TOPS (Int8) performance
- Support for sparse neural networks
- Integration with Windows Studio Effects
- Compatible with major AI frameworks
- Optimized for low-power AI inference

### Performance Characteristics
From our vector addition test with 1 million elements:
- GPU Computation Time: 0.001346 seconds
- CPU Computation Time: 0.001647 seconds
- Achieved Speedup: 1.22x

## Development Environment

### Required Software
- OpenCL Runtime
- OpenCL Headers
- GCC Compiler
- Make build system

### Compilation Flags
```makefile
CFLAGS=-Wall -O3
LDFLAGS=-lOpenCL -lm
```

## Notes

1. The Intel Arc GPU integrated into this processor is fully capable of OpenCL 3.0 computations, making it suitable for general-purpose computing tasks.
2. The system demonstrates the capability to perform parallel computations, though the current simple vector addition test doesn't fully utilize the GPU's potential.
3. For more compute-intensive tasks, the GPU's 64 TOPS (Int8) performance capability could provide significantly better speedups.
4. The dedicated NPU provides an additional 47 TOPS (Int8) of AI processing power, making the total system capable of 115 TOPS when combining CPU, GPU, and NPU capabilities.
5. While our current demo uses the GPU via OpenCL, the NPU could be utilized through frameworks like OpenVINO™ for AI-specific workloads.

## References

[Intel® Core™ Ultra 7 Processor 258V Specifications](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html) 