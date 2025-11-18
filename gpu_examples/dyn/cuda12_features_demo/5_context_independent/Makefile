NVCC = nvcc
CUDA_ARCH = sm_80

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)

all: context_independent_demo fatbin

# Main demo executable
context_independent_demo: context_independent_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda

# Create fatbin for testing
fatbin: test_kernel.fatbin

test_kernel.fatbin: test_kernel.cu
	$(NVCC) $(NVCC_FLAGS) --fatbin $< -o $@

clean:
	rm -f context_independent_demo *.fatbin *.cubin *.ptx

run: all
	./context_independent_demo

help:
	@echo "Context-Independent Module Loading Demo"
	@echo "========================================"
	@echo "Targets:"
	@echo "  make          - Build demo and fatbin"
	@echo "  make fatbin   - Build test fatbin"
	@echo "  make run      - Build and run demo"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  - cuLibrary* context-independent loading"
	@echo "  - Sharing modules across multiple contexts"
	@echo "  - Building a module cache (perfect for CLC!)"
	@echo "  - Memory and performance benefits"

.PHONY: all fatbin clean run help
