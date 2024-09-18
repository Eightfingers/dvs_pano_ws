#include <gtest/gtest.h>
#include "cuda_test/add_parallel.h"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(CudaGPUTest, CMaxGPU)
{
    int N = 100;

    // Run kernel w 256 threads
    // Allocate host memory
    float* h_A = (float*)malloc(N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));
    float* h_C = (float*)malloc(N * sizeof(float));
    
    EXPECT_EQ(test_cuda_(N, h_A, h_B, h_C), true);
    // Free host memory>>
    free(h_A);
    free(h_B);
    free(h_C);
}