#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu_operations.h"
#include "gpu_operations.h"

void runReductionTest(int n, FILE *results_file)
{
    size_t size = n * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(sizeof(float));

    // Initialize with random values between 0 and 1
    for (int i = 0; i < n; i++)
        h_input[i] = (float)rand() / RAND_MAX;

    // CPU Reduction
    double cpu_start = rtclock();
    float cpu_sum = cpuReduce(h_input, n);
    double cpu_end = rtclock();
    double cpu_time = (cpu_end - cpu_start) * 1000.0; // Convert to milliseconds

    // GPU Reduction
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuReduce(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    float error = fabs(cpu_sum - gpu_sum);
    int passed = (error < 1e-3) ? 1 : 0;

    // Write to results file
    fprintf(results_file, "REDUCTION, %d, %.6f, %.6f, %s\n",
            n, cpu_time, gpu_time, passed ? "PASS" : "FAIL");

    printf("Reduction Test (n=%d): CPU=%.3fms, GPU=%.3fms, Error=%.6f, %s\n",
           n, cpu_time, gpu_time, error, passed ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runPrefixSumTest(int n, FILE *results_file)
{
    size_t size = n * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_cpu_result = (float *)malloc(size);
    float *h_gpu_result = (float *)malloc(size);

    // Initialize with random values
    for (int i = 0; i < n; i++)
    {
        h_input[i] = (float)rand() / RAND_MAX;
        h_cpu_result[i] = h_input[i];
    }

    // CPU Prefix Sum
    double cpu_start = rtclock();
    cpuPrefixSum(h_cpu_result, n);
    double cpu_end = rtclock();
    double cpu_time = (cpu_end - cpu_start) * 1000.0;

    // GPU Prefix Sum
    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuPrefixSum(d_input, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    CUDA_CHECK(cudaMemcpy(h_gpu_result, d_input, size, cudaMemcpyDeviceToHost));

    // Verify results
    int passed = 1;
    float max_error = 0;
    for (int i = 0; i < n; i++)
    {
        float error = fabs(h_cpu_result[i] - h_gpu_result[i]);
        if (error > max_error)
            max_error = error;
        if (error > 1e-3)
            passed = 0;
    }

    // Write to results file
    fprintf(results_file, "PREFIX_SUM, %d, %.6f, %.6f, %s\n",
            n, cpu_time, gpu_time, passed ? "PASS" : "FAIL");

    printf("Prefix Sum Test (n=%d): CPU=%.3fms, GPU=%.3fms, MaxError=%.6f, %s\n",
           n, cpu_time, gpu_time, max_error, passed ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_cpu_result);
    free(h_gpu_result);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    // Create results directory
    system("mkdir -p results");

    FILE *results_file = fopen("results/performance_results.csv", "w");
    if (!results_file)
    {
        printf("Error opening results file!\n");
        return -1;
    }

    // Write CSV header
    fprintf(results_file, "Operation, ArraySize, CPUTime(ms), GPUTime(ms), Verification\n");

    // Test cases - different array sizes
    int test_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("=== CUDA Reduction and Prefix Sum Performance Tests ===\n\n");

    // Run reduction tests
    printf("Running Reduction Tests:\n");
    for (int i = 0; i < num_tests; i++)
    {
        runReductionTest(test_sizes[i], results_file);
    }

    printf("\nRunning Prefix Sum Tests:\n");
    // Run prefix sum tests (smaller sizes for prefix sum)
    int prefix_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768};
    int num_prefix_tests = sizeof(prefix_sizes) / sizeof(prefix_sizes[0]);
    for (int i = 0; i < num_prefix_tests; i++)
    {
        runPrefixSumTest(prefix_sizes[i], results_file);
    }

    fclose(results_file);
    printf("\nResults saved to: results/performance_results.csv\n");

    return 0;
}