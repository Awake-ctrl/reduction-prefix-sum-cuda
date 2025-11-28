#include "gpu_operations.h"
#include <stdio.h>

// =================== REDUCTION KERNEL ===================
__global__ void reduceSum(float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = 0;
    if (i < n)
        mySum = input[i];
    if (i + blockDim.x < n)
        mySum += input[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// =================== PREFIX SUM (SCAN) KERNEL ===================
__global__ void prefixSum(float *data, int n)
{
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (n / 2);
    temp[ai] = (ai < n) ? data[ai] : 0;
    temp[bi] = (bi < n) ? data[bi] : 0;

    // Up-sweep (reduce)
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0)
        temp[n - 1] = 0;

    // Down-sweep
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if (ai < n)
        data[ai] = temp[ai];
    if (bi < n)
        data[bi] = temp[bi];
}

void gpuReduce(float *d_input, float *d_output, int n)
{
    int threads = 512;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    size_t smemSize = threads * sizeof(float);

    reduceSum<<<blocks, threads, smemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());

    int s = blocks;
    while (s > 1)
    {
        int nb = (s + threads * 2 - 1) / (threads * 2);
        reduceSum<<<nb, threads, smemSize>>>(d_output, d_output, s);
        CUDA_CHECK(cudaGetLastError());
        s = nb;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpuPrefixSum(float *d_data, int n)
{
    int threads = n / 2;
    size_t smemSize = 2 * n * sizeof(float);

    prefixSum<<<1, threads, smemSize>>>(d_data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}