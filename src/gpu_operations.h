#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                          \
    if (err != cudaSuccess)                                                      \
    {                                                                            \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(-1);                                                                \
    }

void gpuReduce(float *d_input, float *d_output, int n);
void gpuPrefixSum(float *d_data, int n);

#endif