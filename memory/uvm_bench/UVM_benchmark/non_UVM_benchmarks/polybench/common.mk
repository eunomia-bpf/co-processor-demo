all:
	nvcc -O3 -arch=sm_90 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe