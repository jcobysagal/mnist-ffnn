/*
Second attempt at CUDA-Accelerated mnist Feed-Forward Neural Network!
My first attempt worked well with single-batch SGD, fell apart when I tried to parallelize. I decided that I would refactor the code
into a cleaner version now that I have leveled up a bit!
*/

//Includes
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <tuple>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// Headers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// MACROS
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define EPOCHS 30
#define LEARNING_RATE 0.1
#define LAMBDA 0.1
#define BLOCK_SIZE 32


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

// Seed the random number generator
std::random_device rd;
std::mt19937 gen(rd());
// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
// printvec function for debugging
void printvec(float *vec, int sz) {
	for (int i = 0; i < sz; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";
}
// printvec function for debugging
void printvec(int *vec, int sz) {
	for (int i = 0; i < sz; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";
}
// Function to generate batch order for stochastic gradient descent 
std::vector<int> generateShuffledArray(int n, int mini_batch_size) {
    // Create a vector with numbers from 0 to n - 1
    std::vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        numbers[i] = i * mini_batch_size;
    }
    // Shuffle the vector using a random engine
    std::shuffle(numbers.begin(), numbers.end(), gen);
    return numbers;
}
// random number generator from normal distribution
float randnorm() {
	// Define the "standard" normal distribution
	std::normal_distribution<> d(0, 1);
	return d(gen);
}
// get max index of C array
int get_max_index(float* res) {
	int max_index = 0;

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (res[i] > res[max_index]) {
            max_index = i;
        }
	}
	return max_index;
}
// Read mnist training data
std::tuple<int*, float*> read_mnist(std::string a) {

	// Get path to data
	std::filesystem::path current_path = std::filesystem::current_path();

	std::filesystem::path parent_path = current_path.parent_path();

	std::ifstream file;
	int* labels;
    float *data;
	if (a == "test") {
		std::filesystem::path file_path = parent_path / "data/mnist_dataset/mnist_test.csv";
		file.open(file_path);
		labels = (int*)malloc(TEST_SIZE * sizeof(int));
        data = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
	}
	else if (a == "train") {
		std::filesystem::path file_path = parent_path / "data/mnist_dataset/mnist_train.csv";
		file.open(file_path);
		labels = (int*)malloc(TRAIN_SIZE * sizeof(int));
        data = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
	}
	else {
		std::cout << "Invalid input. Please enter 'test' or 'train'." << std::endl;
		return {};
	}
	if (file.is_open()) {
		std::string line;
        int data_i = 0; // counter for data allocation
		int labels_i = 0; // counter for labels allocation
		while (std::getline(file, line)) {
			std::istringstream ss(line);
			std::string token;
			bool label = true; // to normalize everything but the label
			while (std::getline(ss, token, ',')) {
				if (label) {
					labels[labels_i] = std::stoi(token);
					label = false;
                    labels_i++;
				}
				else {
					data[data_i] = std::stof(token) / 255.0;
                    data_i++;
				}
			}
		}
	}
    else {
        std::cout << "Error: Could not open file!" << std::endl;
        return {};
    }
	return std::make_tuple(labels, data);
}
// function that will return tuple (training features, training labels, test features, test labels)
std::tuple<int*, float*, int*, float*> get_data_tuples() {

	std::tuple<int*, float*> training_data;
	std::tuple<int*, float*> test_data;

	std::string a = "train";
	training_data = read_mnist(a);
	a = "test";
	test_data = read_mnist(a);

	return std::make_tuple(std::get<0>(training_data),std::get<1>(training_data),std::get<0>(test_data), std::get<1>(test_data));
}
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
class CUDA_NN{
	public:
	// Constructor
	CUDA_NN(int *sizes, int num_layers, bool large_weights = false) {
		std::cout << "Initializing network..." << std::endl;
		this->num_layers = num_layers;
		this->sizes = sizes;
		this->large_weights = large_weights; 
		// Initialize weights and biases on GPU
		// initialize array of pointers
		weights = (float**)malloc((num_layers-1)*sizeof(float*));
		weights_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		biases = (float**)malloc((num_layers-1)*sizeof(float*));
		biases_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		// Put sizes on GPU
		CUDA_CHECK(cudaMalloc(&sizes_gpu, num_layers * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(sizes_gpu, sizes, num_layers * sizeof(int), cudaMemcpyHostToDevice));
		// Initalize arrays of weights and biases which will contain pointers to the GPU
		for (int i = 0; i < num_layers - 1; i++) {
			int weights_dims = sizes[i] * sizes[i+1];
			int biases_dims = sizes[i+1];

			weights[i] = (float*)malloc(weights_dims * sizeof(float));
			biases[i] = (float*)malloc(biases_dims * sizeof(float));

			initialize_weights(weights[i], i, sizes, large_weights);
			initialize_biases(biases[i], i, sizes);

			CUDA_CHECK(cudaMalloc(&weights_gpu[i], weights_dims * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&biases_gpu[i], biases_dims * sizeof(float)));

			CUDA_CHECK(cudaMemcpy(weights_gpu[i], weights[i], weights_dims * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(biases_gpu[i], biases[i], biases_dims * sizeof(float), cudaMemcpyHostToDevice));
		}
		std::cout << "Initialization complete\n";
	}
	void SGD(int *training_labels, float *training_data, int epochs, int mini_batch_size, float eta, float lmbda, bool val, int *test_labels, float *test_data){
		std::cout << "Beginning stochastic gradient descent" << std::endl;

		float *training_data_gpu, *test_data_gpu;
		int train_size = TRAIN_SIZE * INPUT_SIZE * sizeof(float);
		int test_size = TEST_SIZE * INPUT_SIZE * sizeof(float);

		// Copy data to GPU
		CUDA_CHECK(cudaMalloc(&training_data_gpu, train_size));
		CUDA_CHECK(cudaMemcpy(training_data_gpu, training_data, train_size, cudaMemcpyHostToDevice));
		if (test_data != nullptr) {
			CUDA_CHECK(cudaMalloc(&test_data_gpu, test_size));
			CUDA_CHECK(cudaMemcpy(test_data_gpu, test_data, test_size, cudaMemcpyHostToDevice));
		}
		int num_batches = TRAIN_SIZE/ mini_batch_size;

		for (int epoch = 1; epoch <= EPOCHS; epoch++) {

			std::cout << "Beginning epoch " << epoch << std::endl;
			double st = get_time();

			batch_order = generateShuffledArray(num_batches, mini_batch_size);
			for (int i = 0; i < num_batches; i++) {
				update_mini_batch(batch_order[i], mini_batch_size, training_labels, training_data_gpu, LEARNING_RATE, LAMBDA, TRAIN_SIZE);
			}
			if (test_data != nullptr) {
				std::cout << "Evaluating results" << std::endl;
				std::cout << "Test data accuracy: " << evaluate(test_data_gpu, test_labels) << "/" << TEST_SIZE << " correct." << std::endl;
			}
			double ft = get_time() - st; 
			std::cout << "Epoch " << epoch << " complete." << std::endl;
			std::cout << "Time elapsed = " << ft << " seconds" << std::endl;
		}

		// Free GPU memory
		CUDA_CHECK(cudaFree(training_data_gpu));
		CUDA_CHECK(cudaFree(test_data_gpu));
	}
	// Destructor
	~CUDA_NN(){
		std::cout << "Destructing Neural Network" << std::endl;
		std::cout << "Freeing weights and biases" << std::endl;
		for (int i = 0; i < num_layers - 1; i++){
			free(weights[i]);
			free(biases[i]);
			CUDA_CHECK(cudaFree(weights_gpu[i]));
			CUDA_CHECK(cudaFree(biases_gpu[i]));
		}
		free(weights);
		free(biases);
		free(weights_gpu);
		free(biases_gpu);
	}
	private:
	int *sizes, *sizes_gpu;
	std::vector<int> batch_order;
	int num_layers;
	float **weights;
	float **biases;
	float **weights_gpu;
	float **biases_gpu;
	bool large_weights;

	void initialize_weights(float *weights, int layer, int *sizes, bool large_weights) {

		for (int i = 0; i < sizes[layer]*sizes[layer+1]; i++) {
			weights[i] = randnorm();
			if (!large_weights) {
				weights[i] /= pow(sizes[layer], 0.5);
			}
		}
	}
	void initialize_biases(float *biases, int layer, int *sizes) {
	
		for (int i = 0; i < sizes[layer+1]; i++) { 
			biases[i] = randnorm();
		}
	}
	// Feedforward function
	float* feedforward(float *input_gpu, int mini_batch_size) {
		float **zs_gpu, **activations_gpu;

		zs_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		activations_gpu = (float**)malloc((num_layers-1)*sizeof(float*));

		// Put zs and activations on GPU
		for (int i = 0; i < num_layers - 1; i++){
			CUDA_CHECK(cudaMalloc(&zs_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&activations_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));
		}
		
		dim3 block_size;
		dim3 grid_size;

		// FORWARD PASS
		for (int i = 0; i < num_layers - 1; i++) {
			block_size.x = BLOCK_SIZE;
			block_size.y = BLOCK_SIZE;
			grid_size.x = (sizes[i + 1] + block_size.x - 1) / block_size.x;
			grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;
			if (i == 0) {
				forward<<<grid_size, block_size>>>(input_gpu, weights_gpu[i], biases_gpu[i], zs_gpu[i], activations_gpu[i], sizes[i], sizes[i + 1], mini_batch_size);
			}
			else {
				forward<<<grid_size, block_size>>>(activations_gpu[i-1], weights_gpu[i], biases_gpu[i], zs_gpu[i], activations_gpu[i], sizes[i], sizes[i + 1], mini_batch_size);
			}
		}	
		CUDA_CHECK(cudaDeviceSynchronize());
		// Copy result to output
		float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
		CUDA_CHECK(cudaMemcpy(output, activations_gpu[num_layers - 2], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < num_layers - 1; i++) {
			CUDA_CHECK(cudaFree(zs_gpu[i]));
			CUDA_CHECK(cudaFree(activations_gpu[i]));
		}
		free(activations_gpu);
		free(zs_gpu);
		return output;	
	}
	// function to evaluate performance of the neural network 
	int evaluate(float* test_data, int* test_labels) {
		// Separate test_data into data and labels
		float* res;
		int successes = 0;

		for (int i = 0; i < TEST_SIZE; i++) {
			res = feedforward(&test_data[i * INPUT_SIZE], 1);
			// std::cout << "Printing result for " << i << std::endl;
			// for (int j = 0; j < OUTPUT_SIZE; j++) {
			// 	std::cout << res[j] << " ";
			// }
			// std::cout << std::endl;
			int m = get_max_index(res);
			// std::cout << "Prediction: " << m << ", Label: " << test_labels[i] << std::endl;
			if (m == test_labels[i]) {
				successes++; 
			}
			free(res);
		}
		return successes;
	}
	void update_mini_batch(int batch, int mini_batch_size, int *training_labels, float *training_data, float eta, float lambda, int training_data_size) {
		// backpropagation
		std::vector<float**> nabla = backprop(batch, mini_batch_size, training_labels, training_data); // Tuple contains (nabla_w_gpu, nabla_b_gpu)

		// Update weights and biases
		for (int i = 0; i < num_layers - 1; i++) {
			update_weights<<<(sizes[i] * sizes[i+1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(weights_gpu[i], nabla[0][i], weights_gpu[i], sizes[i] * sizes[i+1], mini_batch_size);
			update_biases<<<(sizes[i+1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(biases_gpu[i], nabla[1][i], biases_gpu[i], sizes[i+1], mini_batch_size);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		for (int i = 0; i < num_layers - 1; i++) {
			CUDA_CHECK(cudaFree(nabla[0][i]));
			CUDA_CHECK(cudaFree(nabla[1][i]));
		}
		free(nabla[0]);
		free(nabla[1]);
	}
	std::vector<float**> backprop(int batch, int mini_batch_size, int *training_labels, float *training_data_gpu) {
		float **nabla_w_gpu;
		float **nabla_b_gpu;
		float **zs_gpu, **activations_gpu, **deltas_gpu;
		float *ys_gpu;

		nabla_w_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		nabla_b_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		zs_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		activations_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		deltas_gpu = (float**)malloc((num_layers-1)*sizeof(float*));
		// Put nablas, zs, activations, and deltas on GPU
		for (int i = 0; i < num_layers - 1; i++){
			CUDA_CHECK(cudaMalloc(&nabla_w_gpu[i], sizes[i]*sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&nabla_b_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&zs_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&activations_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&deltas_gpu[i], sizes[i+1] * mini_batch_size * sizeof(float)));

			// Zero nablas
			CUDA_CHECK(cudaMemset(nabla_w_gpu[i], 0, sizes[i]*sizes[i+1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMemset(nabla_b_gpu[i], 0 , sizes[i+1]  * mini_batch_size * sizeof(float)));
		}

		dim3 block_size;
		dim3 grid_size;
 
		// FORWARD PASS
		for (int i = 0; i < num_layers - 1; i++) {
			block_size.x = BLOCK_SIZE;
			block_size.y = BLOCK_SIZE;
			grid_size.x = (sizes[i + 1] + block_size.x - 1) / block_size.x;
			grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;
			if (i == 0) {
				forward<<<grid_size, block_size>>>
				(&training_data_gpu[batch * INPUT_SIZE], weights_gpu[i], biases_gpu[i], zs_gpu[i], activations_gpu[i], sizes[i], sizes[i + 1], mini_batch_size);
				CUDA_CHECK(cudaGetLastError());
			}
			else {
				forward<<<grid_size, block_size>>>
				(activations_gpu[i-1], weights_gpu[i], biases_gpu[i], zs_gpu[i], activations_gpu[i], sizes[i], sizes[i + 1], mini_batch_size);
				CUDA_CHECK(cudaGetLastError());
			}
		}

		// Vectorize our labels
		float *ys = (float*)malloc(OUTPUT_SIZE * mini_batch_size * sizeof(float));
		memset(ys, 0, OUTPUT_SIZE * mini_batch_size * sizeof(float));
		for (int i = 0; i < mini_batch_size; i++) {
			ys[training_labels[batch + i] + i * OUTPUT_SIZE] = 1.0;
		}
		CUDA_CHECK(cudaMalloc(&ys_gpu, OUTPUT_SIZE * mini_batch_size * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(ys_gpu, ys, OUTPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyHostToDevice));

		grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
		grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;

		matrix_sub<<<grid_size, block_size>>>
		(activations_gpu[num_layers - 2], ys_gpu, deltas_gpu[num_layers - 2], mini_batch_size, OUTPUT_SIZE);
		CUDA_CHECK(cudaGetLastError());

		// Add deltas to nabla_b
		CUDA_CHECK(cudaMemcpy(nabla_b_gpu[num_layers - 2], deltas_gpu[num_layers - 2], OUTPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

		// 3D block
		dim3 block_size_3D(32, 16, 2);
		dim3 grid_size_3D((sizes[num_layers - 2] + block_size_3D.x - 1) / block_size_3D.x, 
		(OUTPUT_SIZE + block_size_3D.y - 1) / block_size_3D.y, 
		(mini_batch_size + block_size_3D.z - 1) / block_size_3D.z);

		// Outer product to get nabla_w
		batch_outer_product<<<grid_size_3D, block_size_3D>>>
		(deltas_gpu[num_layers - 2], activations_gpu[num_layers - 3], nabla_w_gpu[num_layers - 2], OUTPUT_SIZE, sizes[num_layers - 2], mini_batch_size);
		CUDA_CHECK(cudaGetLastError());
		// BACKWARD PASS
		for (int i = num_layers - 2; i > 0; i--) {
			// Calculate sigmoid prime
			sigmoid_prime_vec<<<(sizes[i] * mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(zs_gpu[i - 1], zs_gpu[i - 1], sizes[i] * mini_batch_size);
			CUDA_CHECK(cudaGetLastError());

			grid_size.x = (sizes[i] + block_size.x - 1) / block_size.x;
			matmul_ab<<<grid_size, block_size>>>
			(deltas_gpu[i], weights_gpu[i], deltas_gpu[i - 1], mini_batch_size, sizes[i+1], sizes[i]);
			CUDA_CHECK(cudaGetLastError());

			grid_size.x = (sizes[i] + block_size.x - 1) / block_size.x;
			hadamard_mat<<<grid_size, block_size>>>
			(deltas_gpu[i - 1], zs_gpu[i - 1], deltas_gpu[i - 1], mini_batch_size, sizes[i]);
			CUDA_CHECK(cudaGetLastError());

			// Add deltas to nabla_b
			CUDA_CHECK(cudaMemcpy(nabla_b_gpu[i - 1], deltas_gpu[i - 1], sizes[i] * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

			grid_size_3D.x = ((sizes[i - 1] + block_size_3D.x - 1) / block_size_3D.x);
			grid_size_3D.y = ((sizes[i] + block_size_3D.y - 1) / block_size_3D.y);
			if (i == 1) {
			batch_outer_product<<<grid_size_3D, block_size_3D>>>
			(deltas_gpu[i - 1], &training_data_gpu[batch * INPUT_SIZE], nabla_w_gpu[i - 1], sizes[i], sizes[i - 1], mini_batch_size);	
			CUDA_CHECK(cudaGetLastError());			
			}
			else {
			batch_outer_product<<<grid_size_3D, block_size_3D>>>
			(deltas_gpu[i - 1], activations_gpu[i - 2], nabla_w_gpu[i - 1], sizes[i], sizes[i - 1], mini_batch_size);
			CUDA_CHECK(cudaGetLastError());
			}
		}
		// Loop over nablas and return sums
		float **nabla_w_out, **nabla_b_out;
		nabla_w_out = (float**)malloc((num_layers - 1) * sizeof(float*));
		nabla_b_out = (float**)malloc((num_layers - 1) * sizeof(float*));
		for (int i = 0; i < num_layers - 1; i++) {
			CUDA_CHECK(cudaMalloc(&nabla_w_out[i], sizes[i] * sizes[i + 1] * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMalloc(&nabla_b_out[i], sizes[i + 1] * sizeof(float)));
			CUDA_CHECK(cudaMemset(nabla_w_out[i], 0, sizes[i] * sizes[i + 1] * sizeof(float)));
			CUDA_CHECK(cudaMemset(nabla_b_out[i], 0, sizes[i + 1] * sizeof(float)));
			for (int j = 0; j < mini_batch_size; j++) {
				vector_add<<<(sizes[i] * sizes[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
				(nabla_w_out[i], &nabla_w_gpu[i][j * sizes[i] * sizes[i + 1]], nabla_w_out[i], sizes[i] * sizes[i + 1]);
				vector_add<<<(sizes[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
				(nabla_b_out[i], &nabla_b_gpu[i][j * sizes[i + 1]], nabla_b_out[i], sizes[i + 1]);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaFree(nabla_w_gpu[i]));
			CUDA_CHECK(cudaFree(nabla_b_gpu[i]));
		}
		CUDA_CHECK(cudaDeviceSynchronize());

		// // Print nabla for debugging
		// float **nabla_w_db;
		// nabla_w_db = (float**)malloc((num_layers - 1) * sizeof(float*));
		// for (int i = 0; i < num_layers - 1; i++) {
		// 	std::cout << "Printing weights for layer " << i << "\n";
		// 	nabla_w_db[i] = (float*)malloc(sizes[i] * sizes[i + 1] * sizeof(float));
		// 	CUDA_CHECK(cudaMemcpy(nabla_w_db[i], nabla_w_out[i], sizes[i] * sizes[i + 1] * sizeof(float), cudaMemcpyDeviceToHost));
		//  	printvec(nabla_w_db[i], sizes[i] * sizes[i+1]);
		// 	free(nabla_w_db[i]);
		// }
		// free(nabla_w_db);
		// END Debug

		// Free memory
		for (int i = 0; i < num_layers - 1; i++) {
			CUDA_CHECK(cudaFree(zs_gpu[i]));
			CUDA_CHECK(cudaFree(activations_gpu[i]));
			CUDA_CHECK(cudaFree(deltas_gpu[i]));
		}

		free(nabla_w_gpu);
		free(nabla_b_gpu);
		free(zs_gpu);
		free(activations_gpu);
		free(deltas_gpu);
		free(ys);
		CUDA_CHECK(cudaFree(ys_gpu));

		return std::vector<float**>{nabla_w_out, nabla_b_out};
	}
};
//________________________________MAIN____________________________________
int main() {
 
    std::tuple<int*, float*, int*, float*> data_tuple; 
    std::cout << "Loading MNIST dataset" << std::endl;
    data_tuple = get_data_tuples();
    if (!std::get<0>(data_tuple)) {
        std::cout << "Empty dataset!" << std::endl;
    }
	
	int sizes[] = {INPUT_SIZE, 256, OUTPUT_SIZE};
	int num_layers = sizeof(sizes)/sizeof(int);
	int mini_batch_size = 64;
	CUDA_NN nn(sizes, num_layers);
	nn.SGD(std::get<0>(data_tuple), std::get<1>(data_tuple), EPOCHS, mini_batch_size, LEARNING_RATE, LAMBDA, false, std::get<2>(data_tuple), std::get<3>(data_tuple));

	// Free memory used by mnist data
	free(std::get<0>(data_tuple));
    free(std::get<1>(data_tuple));
	free(std::get<2>(data_tuple));
	free(std::get<3>(data_tuple));

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
} 