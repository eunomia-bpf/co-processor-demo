all:
	nvcc -O3 -arch=sm_90 ${CUFILES} ${DEF} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe