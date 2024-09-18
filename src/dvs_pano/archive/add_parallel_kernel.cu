#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>

#include "cuda_test/add_parallel.h"
#include <cuda_helper.h>

using namespace std::chrono;

inline void safecall(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "CudaError: " << cudaGetErrorString(err) << std::endl;
    }
}

// Multi-threaded Kernel function to add the elements of two arrays
__global__ void add_kernel(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

bool test_cuda_(int N, float *h_A, float *h_B, float *h_C)
{

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc((void**)&d_A, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_B, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy host arrays to device
    safecall(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    safecall(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    auto start = high_resolution_clock::now();
    // Kernel invocation with N threads
    add_kernel<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "Vector addition microseconds: " << duration.count() << std::endl; 

    // Copy result back to host
    safecall(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << std::endl;
            return false;
            break;
        }
    }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return true;
}

__global__ void initVar(float *value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;    
    if (idx < 1){
        device_var = *value;
    }
}

__global__ void getVar(float *d_A2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;    
    if (idx < 1){
        *d_A2 = device_var;
    }
}

void setDeviceVarWrapper_(float *x)
{
    float *d_A;
    safecall(cudaMalloc(&d_A, sizeof(float)));
    safecall(cudaMemcpy(d_A, x, sizeof(float), cudaMemcpyHostToDevice));

    initVar<<<1, 1>>>(d_A);
    cudaDeviceSynchronize();
    safecall(cudaFree(d_A));
}

void getDeviceVarWrapper_(float *x2)
{
    float *d_A2;
    safecall(cudaMalloc(&d_A2, sizeof(float)));
    getVar<<<1, 1>>>(d_A2);
    cudaDeviceSynchronize();

    safecall(cudaMemcpy(x2, d_A2, sizeof(float), cudaMemcpyDeviceToHost));
    safecall(cudaFree(d_A2));
}

void setPositionArrWrapper_(float3 *arr, int n)
{   
    safecall(cudaMemcpyToSymbol(position_arr, arr, n * sizeof(float3), 0, cudaMemcpyHostToDevice));
}

void getPositionArrWrapper_(float3 *arr, int n)
{
    safecall(cudaMemcpyFromSymbol(arr, position_arr, n * sizeof(float3), 0, cudaMemcpyDeviceToHost));
}

__global__ void warpEvents(float3 *d_rotated_ray, float2 *d_warped_pixel_pose, float2 center, float fx, float fy, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = d_rotated_ray[idx].x;
        float y = d_rotated_ray[idx].y;
        float z = d_rotated_ray[idx].z;

        const float phi = atan2f(x,z);
        const float theta = asinf(y / sqrtf(x*x+y*y+z*z));
        d_warped_pixel_pose[idx].x = center.x + (phi * fx);
        d_warped_pixel_pose[idx].y = center.y + (theta * fy);
    }
}

void warpEventsWrapper_(float3 *h_rotated_ray, float2 *h_warped_pixel_pose, float2 center, float fx, float fy, size_t n)
{
    float3 *d_rotated_ray = nullptr;
    float2 * d_warped_pixel_pose = nullptr;

    safecall(cudaMalloc(&d_rotated_ray, n * sizeof(float3)));
    safecall(cudaMemcpy(d_rotated_ray, h_rotated_ray, n * sizeof(float3), cudaMemcpyHostToDevice));

    std::cout << "h_rotated_ray[0]:" <<  h_rotated_ray[0].x << std::endl;
    std::cout << "h_rotated_ray[50]:" <<  h_rotated_ray[50].x << std::endl;
    std::cout << "h_rotated_ray[99]:" <<  h_rotated_ray[99].x << std::endl;


    safecall(cudaMalloc(&d_warped_pixel_pose, n * sizeof(float2)));
    warpEvents<<<(n+255)/256, 256>>>(d_rotated_ray, d_warped_pixel_pose, center, fx, fy, n);
    cudaDeviceSynchronize();
    safecall(cudaMemcpy(h_warped_pixel_pose, d_warped_pixel_pose, n * sizeof(float2), cudaMemcpyDeviceToHost));

    std::cout << "h_warped_pixel_pose[0]:" <<  h_warped_pixel_pose[0].x << std::endl;
    std::cout << "h_warped_pixel_pose[50]:" <<  h_warped_pixel_pose[50].x << std::endl;
    std::cout << "h_warped_pixel_pose[99]:" <<  h_warped_pixel_pose[99].x << std::endl;

    safecall(cudaFree(d_rotated_ray));
    safecall(cudaFree(d_warped_pixel_pose));
}

__global__ void accumulateIL_(float *d_IL, int IL_old_row, int IL_old_cols, int2* d_xy, float2 *d_dxdy, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) 
    {
        int xx = d_xy[idx].x;
        int yy = d_xy[idx].y;
        float dx = d_dxdy[idx].x;
        float dy = d_dxdy[idx].y;

        d_IL[(yy * IL_old_row) + xx             ] += (1.f - dx) * (1.f - dy);
        d_IL[(yy * IL_old_row) + (xx + 1)       ] += dx * (1.f - dy);
        d_IL[(yy + 1) * (IL_old_row + xx)       ] += (1.f - dx) * dy;
        d_IL[(yy + 1) * IL_old_row + (xx + 1) ] += dx * dy;
    }
}

void accumulatePolarityWrapper_(float *h_IL_old, float *h_IL_new, int2 IL_old_dim, int2 IL_new_dim, bool* h_oldevent, int n, int dimension, int number_of_events)
{
    float  *d_IL_old;
    float *d_IL_new;
    bool *d_oldevent;
    std::cout << "Number of pixels of IL panaroma: " <<IL_old_dim.x * IL_old_dim.y << std::endl;
    safecall(cudaMalloc(&d_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float)));
    safecall(cudaMalloc(&d_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float)));
    safecall(cudaMalloc(&d_oldevent, n * sizeof(bool)));

    safecall(cudaMemcpy(d_IL_old, h_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float), cudaMemcpyHostToDevice));
    safecall(cudaMemcpy(d_IL_new, h_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float), cudaMemcpyHostToDevice));
    safecall(cudaMemcpy(d_oldevent, h_oldevent, n * sizeof(bool), cudaMemcpyHostToDevice));
    safecall(cudaGetLastError());
    
    // Calculate how many new and old events there are from the number of events
    int old_counter = 0;
    int new_counter = 0;
    for (int i = 0; i < dimension; i++)
    {
        if (h_oldevent[i] == true)
        {
            old_counter += 1;
        }
        else
        {
            new_counter += 1;
        }
    }

    std::cout << "Total number of new events: " << new_counter << std::endl;
    std::cout << "Total number of old events: " << old_counter << std::endl;

    // For new 
    int2* h_x_y_new = (int2*)malloc(new_counter * sizeof(int2));
    float2* h_dx_dy_new = (float2*)malloc(new_counter * sizeof(float2));
    for (int i = 0; i < new_counter; i++)
    {
        h_x_y_new[i].x = dimension / 2;
        h_x_y_new[i].y = i;
        h_dx_dy_new[i].x = 0.1;
        h_dx_dy_new[i].y = 0.1;
    }
    float2 *d_dx_dy_new;
    safecall(cudaMalloc(&d_dx_dy_new, new_counter * sizeof(float2)));
    safecall(cudaMemcpy(d_dx_dy_new, h_dx_dy_new, new_counter * sizeof(float2), cudaMemcpyHostToDevice));
    int2* d_x_y_new; 
    safecall(cudaMalloc(&d_x_y_new, new_counter * sizeof(int2)));
    safecall(cudaMemcpy(d_x_y_new, h_x_y_new, new_counter * sizeof(int2), cudaMemcpyHostToDevice));

    // For old
    int2* h_x_y_old = (int2*)malloc(old_counter * sizeof(int2));
    float2* h_dx_dy_old = (float2*)malloc(old_counter * sizeof(float2));
    for (int i = 0; i < old_counter; i++)
    {
        h_x_y_old[i].x = dimension / 2;
        h_x_y_old[i].y = i;
        h_dx_dy_old[i].x = 0.1;
        h_dx_dy_old[i].y = 0.1;
    }
    float2 *d_dx_dy_old;
    safecall(cudaMalloc(&d_dx_dy_old, old_counter * sizeof(float2)));
    safecall(cudaMemcpy(d_dx_dy_old, h_dx_dy_old, old_counter * sizeof(float2), cudaMemcpyHostToDevice));
    int2* d_x_y_old; 
    safecall(cudaMalloc(&d_x_y_old, old_counter * sizeof(int2)));
    safecall(cudaMemcpy(d_x_y_old, h_x_y_old, old_counter * sizeof(int2), cudaMemcpyHostToDevice));

    accumulateIL_<<<(old_counter+255)/256, 256>>>(d_IL_old, IL_old_dim.x, IL_old_dim.y, d_x_y_old, d_dx_dy_old, old_counter);

    accumulateIL_<<<(new_counter+255)/256, 256>>>(d_IL_new, IL_old_dim.x, IL_old_dim.y, d_x_y_new, d_dx_dy_new, new_counter);

    cudaDeviceSynchronize();
    safecall(cudaGetLastError());
    safecall(cudaMemcpy(h_IL_old, d_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float), cudaMemcpyDeviceToHost));
    safecall(cudaMemcpy(h_IL_new, d_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float), cudaMemcpyDeviceToHost));

    safecall(cudaFree(d_IL_old));
    safecall(cudaFree(d_IL_new));
    safecall(cudaFree(d_oldevent));
}