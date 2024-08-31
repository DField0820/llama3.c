# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run_fp32.c
	$(CC) -O3 -o run_fp32 run_fp32.c -lm
	$(CC) -O3 -o run_bf16 run_bf16.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run_fp32.c
	$(CC) -g -o run_fp32 run_fp32.c -lm
	$(CC) -g -o run_bf16 run_bf16.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run_fp32.c
	$(CC) -Ofast -o run_fp32 run_fp32.c -lm
	$(CC) -Ofast -o run_bf16 run_bf16.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run_fp32.c
	$(CC) -Ofast -fopenmp -march=native run_fp32.c  -lm  -o run_fp32
	$(CC) -Ofast -fopenmp -march=native run_bf16.c  -lm  -o run_bf16

.PHONY: clean
clean:
	rm -f run_fp32
	rm -f run_bf16
