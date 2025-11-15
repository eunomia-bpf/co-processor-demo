NVCCCFLAGS =  -std=c++17 -lcuda  -arch=sm_90
NVCC = /usr/local/cuda/bin/nvcc

all: 
	$(NVCC) ${NVCCCFLAGS} *.cu -o svm_non_uvm

run:
	./svm_non_uvm  5 1024 93

clean:
	rm -f svm_non_uvm