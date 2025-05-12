# include "nn_kernels.h"

// Functions for ML
// sigmoid function
__device__ float sigmoid(float z) {
	return 1.0 / (1.0 + exp(-z));
}
// sigmoid prime function
__device__ float sigmoid_prime(float z) {
	return sigmoid(z) * (1 - sigmoid(z));
}
// Weights update kernel
__global__ void update_weights(float *a, float *b, float *c, int n, int mini_batch_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = (1 - LEARNING_RATE * (LAMBDA/TRAIN_SIZE)) * a[i] - (LEARNING_RATE/mini_batch_size) * b[i];
	}
}
// Bias update kernel
__global__ void update_biases(float *a, float *b, float *c, int n, int mini_batch_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] - (LEARNING_RATE/mini_batch_size) * b[i];
	}
}
// Forward pass kernel
__global__ void forward(float *input, float *weights, float *biases, float *zs, float *activations, int size_in, int size_out, int mini_batch_size) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < mini_batch_size && col < size_out) {
		// Calculate INPUT * WEIGHTS + BIASES
		float sum = 0.0f;
		for (int i = 0; i < size_in; i++) {
			sum += input[row * size_in + i] * weights[col * size_in + i];
		}
		zs[row * size_out + col] = sum + biases[col];
		// Sigmoid function SIGMOID(Z)
		activations[row * size_out + col] = sigmoid(zs[row * size_out + col]);
	}
}
// Vector add kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}
// Matrix-matrix mult kernel
__global__ void matmul_ab(float *A, float *B, float *C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int l = 0; l < m; l++) {
            sum += A[row * m + l] * B[col + l * k];
        }
        C[row * k + col] = sum;
    }
}
// Batch outer product kernel
__global__ void batch_outer_product(float *A, float *B, float *C, int n, int m, int mini_batch_size){
	// mini_batch_size * n - size of a
	// mini_batch_size * m - size of b
	// n * m - size of c
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int depth = blockIdx.z * blockDim.z + threadIdx.z;
	if (row < n && col < m && depth < mini_batch_size) {
		C[depth * n * m + row * m + col] = A[row + depth * n] * B[col + depth * m];
	}
}
// Matrix sub kernel
__global__ void matrix_sub(float *A, float *B, float *C, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		int idx = row * n + col;
		C[idx] = A[idx] - B[idx];
	}
}
// Sigmoid prime kernel
__global__ void sigmoid_prime_vec(float *a, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		// c[i] = 1.0 / (1.0 + exp(-a[i])) * (1 - 1.0 / (1.0 + exp(-a[i])));
		c[i] = sigmoid_prime(a[i]);
	}
}
// Hadamard product of two matrices kernel
__global__ void hadamard_mat(float *A, float *B, float *C, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		int idx = row * n + col;
		C[idx] = A[idx] * B[idx];
	}
}