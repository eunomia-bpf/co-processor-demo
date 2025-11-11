NVCCCFLAGS =  -std=c++11 -lcuda  -arch=sm_90 -rdc=true
NVCC = /usr/local/cuda/bin/nvcc

all: 
	$(NVCC) ${NVCCCFLAGS} *.cu -o svm_uvm

run:
	./svm_uvm  5 1024 93

clean:
	rm -f svm_uvm