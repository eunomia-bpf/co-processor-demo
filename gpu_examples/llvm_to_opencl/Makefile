CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lOpenCL -lm

# Define paths for LLVM and SPIR-V tools
LLVM_VERSION = 14
SPIRV_LLVM_TOOLS_DIR = $(PWD)/build

all: setup llvm_to_opencl

# Setup LLVM SPIR-V tools
setup:
	@echo "Setting up build environment..."
	@if ! command -v llvm-as-$(LLVM_VERSION) > /dev/null; then \
		echo "Installing LLVM $(LLVM_VERSION)..."; \
		sudo apt-get update && sudo apt-get install -y llvm-$(LLVM_VERSION) llvm-$(LLVM_VERSION)-dev; \
	fi
	@if ! command -v clang-$(LLVM_VERSION) > /dev/null; then \
		echo "Installing Clang $(LLVM_VERSION)..."; \
		sudo apt-get update && sudo apt-get install -y clang-$(LLVM_VERSION); \
	fi
	@mkdir -p $(SPIRV_LLVM_TOOLS_DIR)

# Convert LLVM IR to SPIR-V (for reference only, now done using script)
vector_add.spv: vector_add.ll
	@echo "Converting LLVM IR to SPIR-V..."
	./convert_ir.sh -s -o vector_add.spv vector_add.ll

# Generate OpenCL C code from LLVM IR (for reference only, now done using script)
vector_add_generated.cl: vector_add.ll
	@echo "Generating OpenCL C code from LLVM IR..."
	./convert_ir.sh -c -o vector_add_generated.cl vector_add.ll

# Build the main application
llvm_to_opencl: llvm_to_opencl.c
	$(CC) $(CFLAGS) -o llvm_to_opencl llvm_to_opencl.c $(LDFLAGS)

clean:
	rm -f llvm_to_opencl *.o vector_add.spv generated_kernel.cl vector_add_generated.cl
	rm -rf $(SPIRV_LLVM_TOOLS_DIR)

.PHONY: all setup clean 