# CUDA Reduction and Prefix Sum Implementation

## Project Overview

This project implements efficient parallel versions of reduction and
prefix sum (scan) algorithms using CUDA, with performance comparison
against sequential CPU implementations. The project demonstrates the
significant speedup achievable through GPU parallelization for these
fundamental parallel algorithms.

## Hardware Specification

-   **GPU**: NVIDIA GeForce RTX 2080 Ti\
-   **Compute Capability**: 7.5\
-   **CUDA Cores**: 4352\
-   **Memory**: 11 GB GDDR6\
-   **Memory Bandwidth**: 616 GB/s

## Algorithms Implemented

### 1. Reduction

-   **CPU**: Sequential summation with O(n) time complexity\
-   **GPU**: Tree-based parallel reduction using shared memory with
    O(log n) time complexity

### 2. Prefix Sum (Scan)

-   **CPU**: Sequential cumulative sum with O(n) time complexity\
-   **GPU**: Work-efficient parallel scan (Blelloch algorithm) with
    O(log n) time complexity

## Project Structure

    reduction-prefix-sum-cuda/
    ├── src/
    │   ├── cpu_operations.c
    │   ├── cpu_operations.h
    │   ├── gpu_operations.cu
    │   ├── gpu_operations.h
    │   └── main.cu
    ├── test_cases/
    │   └── generate_test_cases.c
    ├── results/
    ├── bin/
    ├── Makefile
    └── README.md

## Prerequisites

-   NVIDIA GPU with CUDA Compute Capability 3.0 or higher\
-   CUDA Toolkit (version 11.0 or higher recommended)\
-   GCC compiler\
-   Linux environment (tested on Ubuntu 20.04+)

## Installation and Setup

### 1. Create project directory:

``` bash
mkdir reduction-prefix-sum-cuda
cd reduction-prefix-sum-cuda
```

### 2. Create directory structure:

``` bash
mkdir -p src test_cases results bin
```

## Building the Project

### Compile all components:

``` bash
make all
```

### Generate test cases:

``` bash
make tests
```

### Run performance tests:

``` bash
make run
```

## Usage

### Running Complete Test Suite

``` bash
make all
make run
```

### Manual Execution

``` bash
./bin/main
```

### Viewing Results

``` bash
cat results/performance_results.csv
```

## Expected Performance

### Reduction:

-   Small arrays (1K--16K): 5--15× speedup\
-   Medium arrays (64K--256K): 20--40× speedup\
-   Large arrays (1M--4M): 40--60× speedup

### Prefix Sum:

-   Small arrays (1K--8K): 3--10× speedup\
-   Medium arrays (16K--32K): 10--25× speedup

## Output Format

  Column         Description
  -------------- -----------------------------
  Operation      REDUCTION or PREFIX_SUM
  ArraySize      Elements in the array
  CPUTime(ms)    CPU execution time
  GPUTime(ms)    GPU execution time
  Verification   PASS/FAIL correctness check

## Group Members

-   Unnikrishnan\
-   \[Add more\]

## License

This project is for academic purposes.
