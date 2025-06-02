CC=gcc
CFLAGS=-Wall -O3
LDFLAGS=-lOpenCL -lm

all: vector_add

vector_add: vector_add.c
	$(CC) $(CFLAGS) -o vector_add vector_add.c $(LDFLAGS)

clean:
	rm -f vector_add 