#ifndef CPU_OPERATIONS_H
#define CPU_OPERATIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double rtclock();
float cpuReduce(float *data, int n);
void cpuPrefixSum(float *data, int n);

#endif