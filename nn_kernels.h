#ifndef NN_KERNELS
#define NN_KERNELS
// Header file for CUDA Kernels

// Headers
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "mnist_macros.h"


// Modify the CUDA_CHECK macro to print more information
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// Functions for ML
// sigmoid function
__device__ float sigmoid(float z);
// sigmoid prime function
__device__ float sigmoid_prime(float z);
// Weights update kernel
__global__ void update_weights(float *a, float *b, float *c, int n, int mini_batch_size);
// Bias update kernel
__global__ void update_biases(float *a, float *b, float *c, int n, int mini_batch_size);
// Forward pass kernel
__global__ void forward(float *input, float *weights, float *biases, float *zs, float *activations, int size_in, int size_out, int mini_batch_size);
// Vector add kernel
__global__ void vector_add(float *a, float *b, float *c, int n);
// Matrix-matrix mult kernel
__global__ void matmul_ab(float *A, float *B, float *C, int n, int m, int k);
// Batch outer product kernel
__global__ void batch_outer_product(float *A, float *B, float *C, int n, int m, int mini_batch_size);
// Matrix sub kernel
__global__ void matrix_sub(float *A, float *B, float *C, int n, int m);
// Sigmoid prime kernel
__global__ void sigmoid_prime_vec(float *a, float *c, int n);
// Hadamard product of two matrices kernel
__global__ void hadamard_mat(float *A, float *B, float *C, int n, int m);

#endif