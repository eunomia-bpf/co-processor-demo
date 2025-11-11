# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: runq.c
	$(CC) -O3 -o runq -D_FILE_OFFSET_BITS=64 runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
debug: runq.c
	$(CC) -g -o runq -D_FILE_OFFSET_BITS=64 runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: fast
fast: runq.c
	$(CC) -Ofast -o runq -D_FILE_OFFSET_BITS=64 runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./runq out/model.bin
.PHONY: openmp
openmp: runq.c
	$(CC) -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -D_FILE_OFFSET_BITS=64 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu:
	$(CC) -Ofast -std=gnu11 -o runq -D_FILE_OFFSET_BITS=64 runq.c -lm

.PHONY: gnuopenmp
gnuopenmp:
	$(CC) -Ofast -fopenmp -std=gnu11 -D_FILE_OFFSET_BITS=64 runq.c  -lm  -o runq

.PHONY: clean
clean:
	rm -f runq
