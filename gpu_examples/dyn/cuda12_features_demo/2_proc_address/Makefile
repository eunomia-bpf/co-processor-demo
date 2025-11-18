NVCC = nvcc
CUDA_ARCH = sm_80

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH) -O2

all: proc_address_demo

proc_address_demo: proc_address_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda

clean:
	rm -f proc_address_demo

run: proc_address_demo
	./proc_address_demo

help:
	@echo "CUDA 11.3+ cuGetProcAddress Demo"
	@echo "================================="
	@echo "Targets:"
	@echo "  make          - Build demo"
	@echo "  make run      - Build and run demo"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - cuGetProcAddress() to get driver function pointers"
	@echo "  - cudaGetDriverEntryPoint() runtime API access"
	@echo "  - How to build hook/proxy libraries"
	@echo "  - Foundation for CLC policy injection"

.PHONY: all clean run help
