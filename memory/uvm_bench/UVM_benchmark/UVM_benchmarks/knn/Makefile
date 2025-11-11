NVCCCFLAGS =  -std=c++11 -lcuda -O0
NVCC = /usr/local/cuda/bin/nvcc

knn: knn_cuda.cu 
	$(NVCC) ${NVCCCFLAGS} -o $@ $<

clean:
	rm -f knn