#ifndef ADD_PARALLEL_H
#define ADD_PARALLEL_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float device_var;

// Hmm becareful of memcpy n stuff
__constant__ float3 position_arr[10]; // use cudaMemcpyFromSymbol
__device__ float3 *position_arr2; // can only work cudaMemcpy with src from host. if src from device throws compiler warning ???

// Wrapper function to launch the kernel
bool test_cuda_(int N, float *d_A, float *d_B, float *d_C);

void setDeviceVarWrapper_(float *x);
void getDeviceVarWrapper_(float *x2);

void setPositionArrWrapper_(float3 *arr, int n);
void getPositionArrWrapper_(float3 *arr, int n);

void warpEventsWrapper_(float3 *h_rotated_ray, float2 *h_warped_pixel_pose, float2 center, float fx, float fy, size_t n);

void accumulatePolarityWrapper_(float *h_IL_old, float *h_IL_new, int2 IL_old_dim, int2 IL_new_dim, bool* h_oldevent, int n, int dimension, int number_of_events);
#endif // ADD_PARALLEL_H
