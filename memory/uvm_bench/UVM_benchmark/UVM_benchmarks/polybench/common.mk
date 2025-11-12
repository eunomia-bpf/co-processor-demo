all:
	nvcc -O3 --no-device-link -arch=sm_90 ${CUFILES} ${DEF} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe