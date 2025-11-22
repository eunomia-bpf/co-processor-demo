NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CFLAGS = -O3 -arch=sm_120
LDFLAGS = -lcublas

TARGET = cublas_demo
SOURCE = cublas_demo.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
