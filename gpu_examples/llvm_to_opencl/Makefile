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

# Convert LLVM IR to SPIR-V
vector_add.spv: vector_add.ll
	@echo "Converting LLVM IR to SPIR-V..."
	llvm-as-$(LLVM_VERSION) vector_add.ll -o $(SPIRV_LLVM_TOOLS_DIR)/vector_add.bc
	llvm-spirv $(SPIRV_LLVM_TOOLS_DIR)/vector_add.bc -o vector_add.spv

# Alternative: Generate OpenCL C code from LLVM IR via Clang
vector_add_generated.cl: vector_add.ll
	@echo "Generating OpenCL C code from LLVM IR..."
	clang-$(LLVM_VERSION) -S -emit-llvm vector_add.ll -o $(SPIRV_LLVM_TOOLS_DIR)/vector_add.ll
	clang-$(LLVM_VERSION) -x cl -Xclang -finclude-default-header -S -emit-llvm $(SPIRV_LLVM_TOOLS_DIR)/vector_add.ll -o $(SPIRV_LLVM_TOOLS_DIR)/vector_add_cl.ll
	llvm-dis-$(LLVM_VERSION) $(SPIRV_LLVM_TOOLS_DIR)/vector_add_cl.ll -o vector_add_generated.cl

llvm_to_opencl: llvm_to_opencl.c vector_add.spv
	$(CC) $(CFLAGS) -o llvm_to_opencl llvm_to_opencl.c $(LDFLAGS)

clean:
	rm -f llvm_to_opencl *.o vector_add.spv generated_kernel.cl vector_add_generated.cl
	rm -rf $(SPIRV_LLVM_TOOLS_DIR)

.PHONY: all setup clean 