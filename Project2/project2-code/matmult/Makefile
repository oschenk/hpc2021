# On Euler, we will benchmark your DGEMM's performance against the performance
# of the default vendor-tuned DGEMM. This is done in benchmark-blas.
#

CC = gcc
OPT = -O2
CFLAGS = -Wall -std=gnu99 $(OPT) -g
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt -Wl,--no-as-needed -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -m64 -I${MKLROOT}/include

targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-blocked-2 benchmark-blocked-4 benchmark-blocked-8 benchmark-blocked-16 benchmark-blocked-32 benchmark-blocked-64 benchmark-blocked-128
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o  

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
benchmark-blocked-2 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=2 -o $@ $^ $(LDLIBS)
benchmark-blocked-4 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=4 -o $@ $^ $(LDLIBS)
benchmark-blocked-8 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=8 -o $@ $^ $(LDLIBS)
benchmark-blocked-16 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=16 -o $@ $^ $(LDLIBS)
benchmark-blocked-32 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=32 -o $@ $^ $(LDLIBS)
benchmark-blocked-64 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=64 -o $@ $^ $(LDLIBS)
benchmark-blocked-128 : benchmark.o dgemm-blocked.c
	$(CC) $(CFLAGS) -DBLOCKSIZE=128 -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
