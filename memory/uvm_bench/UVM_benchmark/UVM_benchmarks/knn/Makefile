NVCC = /usr/local/cuda/bin/nvcc
NVCCCFLAGS = --no-device-link -arch=sm_90

knn: knn_cuda.cu
	$(NVCC) ${NVCCCFLAGS} $< -o $@

clean:
	rm -f knn *.o