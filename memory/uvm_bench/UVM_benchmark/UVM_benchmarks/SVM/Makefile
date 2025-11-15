NVCCCFLAGS =  -std=c++17 -lcuda  -arch=sm_90
NVCC = /usr/local/cuda/bin/nvcc

all: 
	$(NVCC) ${NVCCCFLAGS} *.cu -o svm_uvm

run:
	./svm_uvm  5 1024 93

clean:
	rm -f svm_uvm