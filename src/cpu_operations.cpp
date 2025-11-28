#include "cpu_operations.h"

double rtclock() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

float cpuReduce(float *data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

void cpuPrefixSum(float *data, int n) {
    for (int i = 1; i < n; i++)
        data[i] += data[i - 1];
}
                                                                                                 
                  