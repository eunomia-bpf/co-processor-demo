CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lOpenCL -lm

all: vector_add matrix_mul llvm_to_opencl

vector_add: vector_add.c
	$(CC) $(CFLAGS) -o vector_add vector_add.c $(LDFLAGS)

matrix_mul: gpu_matrix_mul.c
	$(CC) $(CFLAGS) -o matrix_mul gpu_matrix_mul.c $(LDFLAGS)

llvm_to_opencl:
	$(MAKE) -C llvm_to_opencl

clean:
	rm -f vector_add matrix_mul
	$(MAKE) -C llvm_to_opencl clean 