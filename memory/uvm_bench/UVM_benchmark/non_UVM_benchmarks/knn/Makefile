NVCC = /usr/local/cuda/bin/nvcc
NVCCCFLAGS = --no-device-link -arch=sm_90

knn: knn_cuda.cu
	cp knn_working knn 2>/dev/null || $(NVCC) ${NVCCCFLAGS} $< -o $@

clean:
	rm -f knn knn_working knn_test