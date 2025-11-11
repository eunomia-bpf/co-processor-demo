all:
	nvcc -O3 -arch=sm_90 --no-device-link ${CUFILES} -o ${EXECUTABLE}
clean:
	rm -f *~ *.exe *.o *.fatbin.c *.reg.c *.module_id *.ptx *.cubin *.ii *.cpp* *.stub.c *.gpu