NVCC = nvcc
CUDA_ARCH = sm_120

NVCC_FLAGS = -std=c++17 -arch=$(CUDA_ARCH)
PTX_FLAGS = $(NVCC_FLAGS) --relocatable-device-code=true

all: sample_app extract_and_rewrite policy.ptx

# Target binary (contains kernels to extract)
sample_app: sample_app.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Extraction and rewriting tool
extract_and_rewrite: extract_and_rewrite.cu extractor.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda -lnvJitLink

# Policy PTX
policy.ptx: policy.cu
	$(NVCC) $(PTX_FLAGS) -ptx $< -o $@

# Run the complete demo
run: all
	@echo "========================================="
	@echo "Step 1: Running original application..."
	@echo "========================================="
	./sample_app
	@echo ""
	@echo "========================================="
	@echo "Step 2: Extracting and rewriting..."
	@echo "========================================="
	./extract_and_rewrite ./sample_app

# Just extract (for inspection)
extract: sample_app
	@echo "Extracting kernels from sample_app..."
	cuobjdump -lelf sample_app
	@echo ""
	@echo "Extracting CUBIN files..."
	cuobjdump -xelf all sample_app
	@echo ""
	@echo "Listing symbols..."
	cuobjdump -symbols sample_app

clean:
	rm -f sample_app extract_and_rewrite *.ptx *.cubin

help:
	@echo "Binary Extraction + JIT Rewriting Demo"
	@echo "======================================"
	@echo "Targets:"
	@echo "  make              - Build all components"
	@echo "  make run          - Run complete demo"
	@echo "  make extract      - Extract and inspect kernels"
	@echo "  make clean        - Clean artifacts"
	@echo ""
	@echo "This demo shows:"
	@echo "  1. cuobjdump to extract kernels from binaries"
	@echo "  2. Policy injection into extracted code"
	@echo "  3. JIT rewriting without source code"
	@echo "  4. Real-world binary rewriting workflow"

.PHONY: all run extract clean help
