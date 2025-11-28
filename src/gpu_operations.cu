#include "gpu_operations.h"
#include <stdio.h>

// =================== REDUCTION KERNEL ===================
__global__ void reduceSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load two elements per thread
    float mySum = 0.0f;
    if (i < n) mySum = input[i];
    if (i + blockDim.x < n) mySum += input[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void gpuReduce(float *d_input, float *d_output, int n) {
    int maxThreads = 512;
    int threads = (n < maxThreads * 2) ? (n + 1) / 2 : maxThreads;
    threads = (threads + 31) / 32 * 32; // Make it multiple of 32
    
    if (threads < 32) threads = 32;
    if (threads > 512) threads = 512;
    
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    if (blocks == 0) blocks = 1;
    
    size_t smemSize = threads * sizeof(float);
    
    // Allocate temporary buffer for multiple reductions
    float *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, blocks * sizeof(float)));
    
    // First reduction
    reduceSum<<<blocks, threads, smemSize>>>(d_input, d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Subsequent reductions if needed
    int elements = blocks;
    while (elements > 1) {
        int new_blocks = (elements + threads * 2 - 1) / (threads * 2);
        if (new_blocks == 0) new_blocks = 1;
        
        reduceSum<<<new_blocks, threads, smemSize>>>(d_temp, d_temp, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        elements = new_blocks;
    }
    
    // Copy final result
    if (elements == 1) {
        CUDA_CHECK(cudaMemcpy(d_output, d_temp, sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =================== OPTIMIZED PARALLEL PREFIX SUM ===================

// Work-efficient parallel scan (Blelloch algorithm) - EXCLUSIVE SCAN
__global__ void parallelScan(float *data, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    int ai = tid;
    int bi = tid + (n/2);
    
    // Handle cases where n is not power of 2
    temp[ai] = (ai < n) ? data[ai] : 0;
    temp[bi] = (bi < n) ? data[bi] : 0;
    
    __syncthreads();
    
    // Build sum in place up the tree (up-sweep)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }
    
    // Clear the last element for exclusive scan
    if (tid == 0 && n > 0) {
        temp[n - 1] = 0;
    }
    
    // Traverse down tree & build scan (down-sweep)
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            float t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();
    
    // Write results to device memory
    if (ai < n) data[ai] = temp[ai];
    if (bi < n) data[bi] = temp[bi];
}

// Convert exclusive scan to inclusive scan
__global__ void exclusiveToInclusive(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n - 1) {
        data[tid] = data[tid + 1];
    }
    if (tid == n - 1) {
        // Last element: sum of all elements
        data[tid] = data[tid] + ((tid > 0) ? data[0] : 0);
    }
}

// Simple inclusive scan for verification
__global__ void simpleInclusiveScan(float *data, int n) {
    int tid = threadIdx.x;
    
    for (int stride = 1; stride < n; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            data[tid] += data[tid - stride];
        }
        __syncthreads();
    }
}

// Multi-block scan for large arrays
__global__ void blockScan(float *data, float *block_sums, int n, int block_size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = bid * block_size;
    
    // Load data into shared memory
    int idx = start + tid;
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    // Scan within block (inclusive)
    for (int stride = 1; stride < block_size; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            sdata[tid] += sdata[tid - stride];
        }
        __syncthreads();
    }
    
    // Store block sum and write results
    if (tid == block_size - 1 && (start + block_size - 1) < n) {
        block_sums[bid] = sdata[tid];
    }
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Add block sums to scanned blocks
__global__ void addBlockSums(float *data, float *block_sums, int n, int block_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = bid * block_size + tid;
    
    if (bid > 0 && start < n) { // Skip first block
        data[start] += block_sums[bid - 1];
    }
}

void gpuPrefixSum(float *d_data, int n) {
    // For very small arrays, use simple inclusive scan
    if (n <= 512) {
        int threads = n;
        if (threads > 512) threads = 512;
        
        // Use simple inclusive scan for small arrays
        simpleInclusiveScan<<<1, threads>>>(d_data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    
    // For medium arrays, use work-efficient exclusive scan + conversion
    if (n <= 2048) {
        int threads = (n + 1) / 2;
        if (threads > 1024) threads = 1024;
        
        size_t smemSize = n * sizeof(float);
        
        // First do exclusive scan
        parallelScan<<<1, threads, smemSize>>>(d_data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Convert to inclusive scan
        int convert_threads = 256;
        int convert_blocks = (n + convert_threads - 1) / convert_threads;
        exclusiveToInclusive<<<convert_blocks, convert_threads>>>(d_data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    
    // For large arrays, use multi-block approach
    int block_size = 512;
    int num_blocks = (n + block_size - 1) / block_size;
    
    // Allocate memory for block sums
    float *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));
    
    // Step 1: Scan each block (inclusive scan within blocks)
    size_t smemSize = block_size * sizeof(float);
    blockScan<<<num_blocks, block_size, smemSize>>>(d_data, d_block_sums, n, block_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Make block sums exclusive by shifting
    if (num_blocks > 1) {
        // Store last element of each block sum for later
        float *d_block_sums_temp;
        CUDA_CHECK(cudaMalloc(&d_block_sums_temp, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_block_sums_temp, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Scan the block sums (exclusive)
        gpuPrefixSum(d_block_sums_temp, num_blocks);
        
        // Copy scanned block sums back
        CUDA_CHECK(cudaMemcpy(d_block_sums, d_block_sums_temp, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_block_sums_temp));
    } else {
        // Single block case
        float zero = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_block_sums, &zero, sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Step 3: Add scanned block sums to each block
    addBlockSums<<<num_blocks, block_size>>>(d_data, d_block_sums, n, block_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_block_sums));
}