simpush: SFMT.c main.cpp kernel.cu
	nvcc -std=c++11  -gencode arch=compute_35,code=sm_35 -DSFMT_MEXP=607 -I SFMT-src-1.4.1/ -I ./cub -I. -O3 -w -o simpush SFMT.c main.cpp kernel.cu -lcudart -lcuda -lcurand
