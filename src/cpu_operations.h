#ifndef CPU_OPERATIONS_H
#define CPU_OPERATIONS_H

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern "C" {
    double rtclock();
    float cpuReduce(float *data, int n);
    void cpuPrefixSum(float *data, int n);
}

#endif
