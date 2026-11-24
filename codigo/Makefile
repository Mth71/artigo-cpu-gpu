# Compilação dos benchmarks.
#   make          -> mm_cpu (com OpenBLAS) e mm_gpu
#   make cpu      -> só mm_cpu
#   make gpu      -> só mm_gpu
#   make OPENBLAS=0 cpu   -> mm_cpu sem OpenBLAS
#   make SM=86 gpu        -> fixa a arquitetura (86 = RTX 30xx, 89 = RTX 40xx, 75 = GTX 16xx/RTX 20xx)

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  := -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
OPENBLAS ?= 1
SM      ?= native

ifeq ($(OPENBLAS),1)
  CFLAGS += -DUSE_OPENBLAS -I/usr/include/x86_64-linux-gnu/openblas-pthread
  LDBLAS := -lopenblas
endif

ifeq ($(SM),native)
  ARCH := -arch=native
else
  ARCH := -gencode arch=compute_$(SM),code=sm_$(SM)
endif

all: cpu gpu
cpu: mm_cpu
gpu: mm_gpu

mm_cpu: mm_cpu.c comum.h
	$(CC) $(CFLAGS) -o $@ mm_cpu.c $(LDBLAS) -lm

mm_gpu: mm_gpu.cu comum.h
	$(NVCC) -O3 $(ARCH) -o $@ mm_gpu.cu -lcublas

clean:
	rm -f mm_cpu mm_gpu

.PHONY: all cpu gpu clean
