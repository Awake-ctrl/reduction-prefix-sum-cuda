#include "cpu_operations.h"

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float cpuReduce(float *data, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

void cpuPrefixSum(float *data, int n)
{
    for (int i = 1; i < n; i++)
        data[i] += data[i - 1];
}