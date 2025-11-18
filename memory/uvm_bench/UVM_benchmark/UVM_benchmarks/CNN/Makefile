include ../../common/make.config


CC := $(CUDA_DIR)/bin/nvcc

all:
	$(CC) -arch=sm_90 -I$(CUDA_DIR)/targets/x86_64-linux/include -L$(CUDA_DIR)/targets/x86_64-linux/lib -lcuda -lcublas *.cu -o CNN

run:
	./CNN
clean:
	rm -f CNN
