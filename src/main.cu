#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cpu_operations.h"
#include "gpu_operations.h"

void debugPrefixSum(int n) {
    printf("\n=== Debugging Prefix Sum for n=%d ===\n", n);
    
    size_t size = n * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_cpu_result = (float*)malloc(size);
    float *h_gpu_result = (float*)malloc(size);

    // Initialize with simple pattern
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
        h_cpu_result[i] = h_input[i];
        h_gpu_result[i] = h_input[i];
    }

    // CPU Prefix Sum
    cpuPrefixSum(h_cpu_result, n);
    
    printf("CPU Result (first 10): ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.1f ", h_cpu_result[i]);
    }
    printf("\n");

    // GPU Prefix Sum
    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    gpuPrefixSum(d_input, n);
    
    CUDA_CHECK(cudaMemcpy(h_gpu_result, d_input, size, cudaMemcpyDeviceToHost));
    
    printf("GPU Result (first 10): ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.1f ", h_gpu_result[i]);
    }
    printf("\n");

    // Verify
    int correct = 1;
    for (int i = 0; i < n; i++) {
        float expected = (float)(i + 1);
        if (fabs(h_gpu_result[i] - expected) > 1e-3) {
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected, h_gpu_result[i]);
            correct = 0;
            if (i >= 10) break; // Show first 10 errors only
        }
    }
    
    if (correct) {
        printf("✓ Prefix Sum CORRECT!\n");
    } else {
        printf("✗ Prefix Sum INCORRECT!\n");
    }

    free(h_input);
    free(h_cpu_result);
    free(h_gpu_result);
    cudaFree(d_input);
}

void runReductionTest(int n, FILE *results_file) {
    size_t size = n * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(sizeof(float));
    
    // Initialize with predictable values for debugging
    for (int i = 0; i < n; i++) 
        h_input[i] = 1.0f;  // All ones for easy verification

    // CPU Reduction
    double cpu_start = rtclock();
    float cpu_sum = cpuReduce(h_input, n);
    double cpu_end = rtclock();
    double cpu_time = (cpu_end - cpu_start) * 1000.0;

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

    // Verify results - should be exactly n for all ones
    float expected = (float)n;
    float cpu_error = fabs(cpu_sum - expected);
    float gpu_error = fabs(gpu_sum - expected);
    int passed = (gpu_error < 1e-3) ? 1 : 0;

    // Write to results file
    fprintf(results_file, "REDUCTION, %d, %.6f, %.6f, %s\n",
            n, cpu_time, gpu_time, passed ? "PASS" : "FAIL");

    printf("Reduction Test (n=%d):\n", n);
    printf("  Expected: %.1f, CPU: %.1f, GPU: %.1f\n", expected, cpu_sum, gpu_sum);
    printf("  CPU Error: %.6f, GPU Error: %.6f\n", cpu_error, gpu_error);
    printf("  CPU Time: %.3fms, GPU Time: %.3fms, %s\n", 
           cpu_time, gpu_time, passed ? "PASS" : "FAIL");
    printf("  Speedup: %.2fx\n\n", cpu_time / gpu_time);

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runPrefixSumTest(int n, FILE *results_file) {
    size_t size = n * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_cpu_result = (float*)malloc(size);
    float *h_gpu_result = (float*)malloc(size);

    // Initialize with simple pattern (all ones for easy verification)
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All ones - prefix sum should be [1, 2, 3, ..., n]
        h_cpu_result[i] = h_input[i];
        h_gpu_result[i] = h_input[i];
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

    // Verify results - prefix sum of all ones should be [1, 2, 3, ..., n]
    int passed = 1;
    float max_error = 0;
    int error_count = 0;
    
    for (int i = 0; i < n; i++) {
        float expected = (float)(i + 1);
        float gpu_val = h_gpu_result[i];
        float error = fabs(gpu_val - expected);
        
        if (error > max_error) max_error = error;
        if (error > 1e-3) {
            passed = 0;
            error_count++;
            if (error_count <= 5) { // Print first 5 errors for debugging
                printf("  Index %d: Expected %.1f, GPU got %.6f, Error: %.6f\n", 
                       i, expected, gpu_val, error);
            }
        }
    }

    if (error_count > 5) {
        printf("  ... and %d more errors\n", error_count - 5);
    }

    // Write to results file
    fprintf(results_file, "PREFIX_SUM, %d, %.6f, %.6f, %s\n",
            n, cpu_time, gpu_time, passed ? "PASS" : "FAIL");

    printf("Prefix Sum Test (n=%d): CPU=%.3fms, GPU=%.3fms, MaxError=%.6f, %s\n",
           n, cpu_time, gpu_time, max_error, passed ? "PASS" : "FAIL");
    if (gpu_time > 0) {
        printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    } else {
        printf("  Speedup: N/A (GPU time too small)\n");
    }
    printf("\n");

    // Cleanup
    free(h_input);
    free(h_cpu_result);
    free(h_gpu_result);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Create results directory
    int result = system("mkdir -p results");
    (void)result; // Suppress unused warning

    FILE *results_file = fopen("results/performance_results.csv", "w");
    if (!results_file) {
        printf("Error opening results file!\n");
        return -1;
    }

    // Write CSV header
    fprintf(results_file, "Operation, ArraySize, CPUTime(ms), GPUTime(ms), Verification\n");

    printf("=== CUDA Reduction and Prefix Sum Performance Tests ===\n\n");

    // First debug small prefix sums to ensure they work
    debugPrefixSum(8);
    debugPrefixSum(16);
    debugPrefixSum(32);

    // Test cases - different array sizes
    int reduction_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    int num_reduction_tests = sizeof(reduction_sizes) / sizeof(reduction_sizes[0]);

    // Run reduction tests
    printf("Running Reduction Tests:\n");
    for (int i = 0; i < num_reduction_tests; i++) {
        runReductionTest(reduction_sizes[i], results_file);
    }

    printf("\nRunning Prefix Sum Tests:\n");
    // Test both small and large arrays
    int prefix_sizes[] = {1024, 2048, 4096, 8192};
    int num_prefix_tests = sizeof(prefix_sizes) / sizeof(prefix_sizes[0]);
    for (int i = 0; i < num_prefix_tests; i++) {
        runPrefixSumTest(prefix_sizes[i], results_file);
    }

    fclose(results_file);
    printf("\n=== All Tests Completed ===\n");
    printf("Results saved to: results/performance_results.csv\n");

    return 0;
}