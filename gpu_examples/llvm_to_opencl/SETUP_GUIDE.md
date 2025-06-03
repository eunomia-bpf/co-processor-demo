# LLVM IR to SPIR-V Conversion Setup Guide

This guide provides detailed instructions for setting up a complete environment for converting LLVM IR code to SPIR-V, which can then be executed on OpenCL-compatible devices.

## Prerequisites

To successfully convert LLVM IR to SPIR-V, you'll need the following components:

1. LLVM toolchain (with a version compatible with your IR files)
2. SPIRV-LLVM-Translator
3. SPIRV-Tools (optional, for validation and assembly/disassembly)
4. OpenCL runtime and development libraries

## Step 1: Install Compatible LLVM Toolchain

The example in this repository uses LLVM IR with typed pointers. If you have a newer version of LLVM (LLVM 14+), you might face issues with opaque pointers, as they became the default in newer LLVM versions.

### Option 1: Install LLVM 14 (Last version with typed pointers by default)

For Ubuntu/Debian:
```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-14 main"
sudo apt-get update
sudo apt-get install llvm-14 llvm-14-dev clang-14 libclang-14-dev
```

For other distributions, refer to [LLVM's download page](https://releases.llvm.org/download.html).

### Option 2: Use Newer LLVM with Typed Pointer Option

If you're using a newer LLVM version, you can use the `-opaque-pointers=0` flag:
```bash
llvm-as -opaque-pointers=0 input.ll -o output.bc
```

## Step 2: Build SPIRV-LLVM-Translator

The SPIRV-LLVM-Translator is essential for converting LLVM bitcode to SPIR-V:

```bash
# Clone the repository with a tag matching your LLVM version
git clone -b llvm_release_140 https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
mkdir build && cd build

# Configure with your LLVM installation
cmake -DLLVM_DIR=/usr/lib/llvm-14/lib/cmake/llvm/ ..

# Build
make -j$(nproc)
```

After building, the `llvm-spirv` tool will be available in `build/tools/llvm-spirv/`.

## Step 3: Install SPIRV-Tools (Optional)

SPIRV-Tools provide utilities for working with SPIR-V files:

```bash
git clone https://github.com/KhronosGroup/SPIRV-Tools.git
cd SPIRV-Tools
git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## Step 4: Install OpenCL Development Libraries

For Ubuntu/Debian:
```bash
sudo apt-get install ocl-icd-opencl-dev
```

For other distributions, refer to your package manager or the OpenCL vendor's documentation.

## Converting LLVM IR to SPIR-V

Once you have all the components installed, you can convert your LLVM IR file to SPIR-V:

```bash
# Step 1: Convert LLVM IR to bitcode
llvm-as-14 input.ll -o input.bc

# Step 2: Convert bitcode to SPIR-V
llvm-spirv input.bc -o output.spv
```

If you're using a newer LLVM version with opaque pointers:
```bash
llvm-as-14 -opaque-pointers=0 input.ll -o input.bc
llvm-spirv input.bc -o output.spv
```

## Troubleshooting

### Opaque Pointer Issues

If you encounter errors like:
```
Opaque pointers are only supported in -opaque-pointers mode
```

Try using a version of LLVM that matches your IR file's format (LLVM 14 or earlier for typed pointers) or use the `-opaque-pointers=0` flag with newer LLVM versions.

### SPIR-V Validation Issues

If your SPIR-V file doesn't work with OpenCL, validate it with SPIRV-Tools:
```bash
spirv-val output.spv
```

This will help identify any issues in the SPIR-V binary.

### OpenCL Compatibility

Not all OpenCL implementations support SPIR-V equally. Check your device's OpenCL extensions to verify SPIR-V support:
```bash
clinfo | grep -i spir
```

Look for `cl_khr_il_program` in the supported extensions.

## Using the Script in This Repository

The `convert_ir.sh` script in this repository automates the conversion process:

```bash
# Convert to SPIR-V
./convert_ir.sh -s -o output.spv input.ll

# Convert to OpenCL C (if possible)
./convert_ir.sh -c -o output.cl input.ll
```

Make sure to have the required tools in your PATH or update the script to point to your installations.

## Additional Resources

- [LLVM Documentation](https://llvm.org/docs/)
- [SPIRV-LLVM-Translator Documentation](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
- [SPIR-V Specification](https://registry.khronos.org/SPIR-V/)
- [OpenCL Documentation](https://www.khronos.org/opencl/) 