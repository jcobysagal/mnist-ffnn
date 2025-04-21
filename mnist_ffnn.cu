/*
First attempt at CUDA-Accelerated mnist Feed-Forward Neural Network!
Porting from my C++ code, which was my first attempt at a from-scratch neural network in the language.
This will include several refactors of that code mostly because I will have to put everything into 1D arrays instead of nested vectors to make
data management easier for CUDA.
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

// MACROS
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define EPOCHS 30
#define LEARNING_RATE 3.0
#define LAMBDA 0
#define BLOCK_SIZE 256


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
// Function to generate batch order for stochastic gradient descent 
std::vector<int> generateShuffledArray(int n) {
    // Create a vector with numbers from 0 to n - 1
    std::vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        numbers[i] = i;
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
		labels = new int[TEST_SIZE];
        data = new float[TEST_SIZE * INPUT_SIZE];
	}
	else if (a == "train") {
		std::filesystem::path file_path = parent_path / "data/mnist_dataset/mnist_train.csv";
		file.open(file_path);
		labels = new int[TRAIN_SIZE];
        data = new float[TRAIN_SIZE * INPUT_SIZE];
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
// Define our CUDA kernels (for now it's all naive approach)
// Sigmoid kernel
__global__ void sigmoid_vec(float *a, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		// c[i] = 1.0 / (1.0 + exp(-a[i]));
		c[i] = sigmoid(a[i]);
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
// Vector add kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}
// Add vector to matrix kernel
__global__ void add_vec_to_mat(float *A, float *b, float *C, int n, int m) {
	// n by m matrix A
	// vector size m b
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < m) {
		C[row * m + col] = A[row * m + col] + b[col];
	}
}
// Vector add kernel
__global__ void vector_sub(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] - b[i];
	}
}
// Vector outer product kernel
__global__ void outer_product(float *a, float *b, float *c, int n, int m) {
	// n - size of a
	// m - size of b
	// n * m - size of c
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && j < m) {
		c[i * m + j] = a[i] * b[j];
	}
}
// Batch outer product kernel
__global__ void batch_outer_product(float *A, float *B, float *C, int n, int m, int mini_batch_size, int weights_len){
	// mini_batch_size * n - size of a
	// mini_batch_size * m - size of b
	// n * m - size of c
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int depth = blockIdx.z * blockDim.z + threadIdx.z;
	if (row < n && col < m && depth < mini_batch_size) {
		C[depth * weights_len + row * m + col] = A[row + depth * n] * B[col + depth * m];
	}
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
// Matrix add kernel
__global__ void matrix_add(float *A, float *B, float *C, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		int idx = row * n + col;
		C[idx] = A[idx] + B[idx];
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
// Matrix-vector mult kernel
__global__ void matvecmul_gpu(float *A, float *b, float *c, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            sum += A[row * k + j] * b[j];
        }
        c[row] = sum;
    }
}
// Matrix_T-vector mult kernel
__global__ void matTvecmul_gpu(float *A, float *b, float *c, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < k) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            sum += A[row +  k * j] * b[j];
        }
        c[row] = sum;
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
// MatrixT-matrix mult kernel
__global__ void matmul_aTb(float *A, float *B, float *C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int l = 0; l < n; l++) {
            sum += A[row + m * l] * B[col + l * k];
        }
        C[row * k + col] = sum;
    }
}
// Matrix-matrixT mult kernel
__global__ void matmul_abT(float *A, float *B, float *C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[col * k + l];
        }
        C[row * m + col] = sum;
    }
}
// Hadamard product of two vectors kernel
__global__ void hadamard_vec(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] * b[i];
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
	CUDA_NN(std::vector<int>& sizes, bool large_weights = false) {
		std::cout << "Initializing network..." << std::endl;
		num_layers = sizes.size();
		this->sizes = sizes;
		this->large_weights = large_weights;

		// Get total number of weights
		weights_len = 0;
		for (int i = 0; i < num_layers - 1; i++) {
			weights_len += sizes[i]*sizes[i+1];
		}
		// Get total number of biases
		biases_len = 0;
		for (int i = 1; i < num_layers; i++) {
			biases_len += sizes[i];
		}
		// Initialize weights and biases
		weights = new float[weights_len]();
		biases = new float[biases_len]();
		initialize_weights();
		initialize_biases();
		size_w = weights_len * sizeof(float);
		size_b = biases_len * sizeof(float);
	}
	// Destructor
	~CUDA_NN(){
		std::cout << "Destructing Neural Network" << std::endl;
		std::cout << "Freeing weights and biases" << std::endl;
		delete[] weights;
		delete[] biases;
	}

	void SGD(int *training_labels, float *training_data, int epochs, int mini_batch_size, float eta, float lmbda, bool val, int *test_labels, float *test_data) {
		std::cout << "Beginning stochastic gradient descent" << std::endl;

		float *training_data_gpu, *test_data_gpu;
		int train_size = TRAIN_SIZE * INPUT_SIZE * sizeof(float);
		int test_size = TEST_SIZE * INPUT_SIZE * sizeof(float);

		// Allocate Weights and Biases on GPU
		CUDA_CHECK(cudaMalloc(&weights_gpu, size_w));
		CUDA_CHECK(cudaMalloc(&biases_gpu, size_b));
		CUDA_CHECK(cudaMemcpy(weights_gpu, weights, size_w, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(biases_gpu, biases, size_b, cudaMemcpyHostToDevice));

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
			batch_order = generateShuffledArray(num_batches);
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
		// Copy data from GPU
		CUDA_CHECK(cudaMemcpy(weights, weights_gpu, size_w, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(biases, biases_gpu, size_b, cudaMemcpyDeviceToHost));

		// Free GPU memory
		CUDA_CHECK(cudaFree(training_data_gpu));
		CUDA_CHECK(cudaFree(test_data_gpu));
		CUDA_CHECK(cudaFree(weights_gpu));
		CUDA_CHECK(cudaFree(biases_gpu));
	}

	private:
	std::vector<int> sizes;
	std::vector<int> batch_order;
	float *weights, *biases;
	float *weights_gpu, *biases_gpu;
	size_t num_layers;
	int weights_len, biases_len, size_w, size_b;
	bool large_weights;
	
	void initialize_weights() {
		int accum = 0;
		for (int i = 0; i < num_layers - 1; i++) {
			int j = 0;
			while (j < accum + sizes[i]*sizes[i+1]) {
				weights[j] = randnorm();
				if (!large_weights) {
					weights[j] /= pow(sizes[i], 0.5);
				}
				j++;
			}
			accum += sizes[i]*sizes[i+1];
		}	
	}
	void initialize_biases() {
		for (int i = 0; i < biases_len; i++) {
			biases[i] = randnorm();
		}
	}
	// Feedforward function
	float* feedforward(float *input_gpu) {
		float *output = new float[OUTPUT_SIZE];
		float *activations_gpu, *zs_gpu;

		int activations_len = std::accumulate(sizes.begin(), sizes.end(), 0);
		int zs_len = std::accumulate(sizes.begin() + 1, sizes.end(), 0);
		int size_activations = activations_len * sizeof(float);
		int size_zs = zs_len * sizeof(float);

		// We want to make some ints of accumulated weights, biases and activations to make the next part easier
		int accum_w = 0, accum_z = 0, accum_a = 0; // We can use z and b interchangeably for this
		// Put input, activations, and zs on GPU
		CUDA_CHECK(cudaMalloc(&activations_gpu, size_activations));
		CUDA_CHECK(cudaMemset(activations_gpu, 0, size_activations));
		CUDA_CHECK(cudaMalloc(&zs_gpu, size_zs));
		// Set first TRAIN_SIZE activations to input
		CUDA_CHECK(cudaMemcpy(activations_gpu, input_gpu, INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

		// Forward pass
		for (int i = 0; i < num_layers - 1; i++) {
			// Calculate z[i] = w[i] * a[i-1] + b[i]
			matvecmul_gpu<<<(sizes[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(&weights_gpu[accum_w], &activations_gpu[accum_a], &zs_gpu[accum_z], sizes[i + 1], sizes[i]);

			accum_a += sizes[i]; // updated accumulated A layers

			vector_add<<<(sizes[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(&zs_gpu[accum_z], &biases_gpu[accum_z], &zs_gpu[accum_z], sizes[i + 1]);

			// Calculate a[z[i]]
			sigmoid_vec<<<(sizes[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(&zs_gpu[accum_z], &activations_gpu[accum_a], sizes[i + 1]);


			if (i < num_layers - 2) {
				accum_w += sizes[i] * sizes[i + 1];
				accum_z += sizes[i + 1];
			}
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		// Copy result to output
		CUDA_CHECK(cudaMemcpy(output, &activations_gpu[accum_a], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(activations_gpu));
		CUDA_CHECK(cudaFree(zs_gpu));

		return output;
	}

	//function to evaluate performance of the neural network 
	int evaluate(float* test_data, int* test_labels) {
		// Separate test_data into data and labels
		float* res;
		int successes = 0;

		for (int i = 0; i < TEST_SIZE; i++) {
			res = feedforward(&test_data[i * INPUT_SIZE]);
			// std::cout << "Printing result for " << i << std::endl;
			// for (int j = 0; j < OUTPUT_SIZE; j++) {
			// 	std::cout << res[j] << " ";
			// }
			// std::cout << std::endl;
			int m = get_max_index(res);
			if (m == test_labels[i]) {
				successes++;
			}
			delete[] res;
		}
		return successes;
	}

	void update_mini_batch(int batch, int mini_batch_size, int *training_labels, float *training_data, float eta, float lambda, int training_data_size) {
		float *nabla_w = new float[weights_len]();
		float *nabla_b = new float[biases_len]();
		float *nabla_w_gpu, *nabla_b_gpu;

		CUDA_CHECK(cudaMalloc(&nabla_w_gpu, size_w));
		CUDA_CHECK(cudaMalloc(&nabla_b_gpu, size_b));
		CUDA_CHECK(cudaMemcpy(nabla_w_gpu, nabla_w, size_w, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(nabla_b_gpu, nabla_b, size_b, cudaMemcpyHostToDevice));
		// backpropagation
		std::tuple<float*, float*> delta_nabla = backprop(batch, mini_batch_size, training_labels, training_data); // Tuple contains pointers to GPU data (nabla_w_gpu, nabla_b_gpu)

		// vector_add<<<(weights_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nabla_w_gpu, std::get<0>(delta_nabla), nabla_w_gpu, weights_len);
		// vector_add<<<(biases_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nabla_b_gpu, std::get<1>(delta_nabla), nabla_b_gpu, biases_len);
		CUDA_CHECK(cudaMemcpy(nabla_w_gpu, std::get<0>(delta_nabla), size_w, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(nabla_b_gpu, std::get<1>(delta_nabla), size_b, cudaMemcpyDeviceToDevice));
		// CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaFree(std::get<0>(delta_nabla)));
		CUDA_CHECK(cudaFree(std::get<1>(delta_nabla)));
		// Create dummy weights for testing
		// float *dummy_w = new float[weights_len];
		// CUDA_CHECK(cudaMemcpy(dummy_w, nabla_w_gpu, size_w, cudaMemcpyDeviceToHost));
		// for (int i = INPUT_SIZE * 30; i < weights_len; i++) {
		// 	std::cout <<  dummy_w[i] << " ";
		// }
		// std::cout << std::endl;

		// Update weights and biases
		update_weights<<<(weights_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(weights_gpu, nabla_w_gpu, weights_gpu, weights_len, mini_batch_size);
		update_biases<<<(biases_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(biases_gpu, nabla_b_gpu, biases_gpu, biases_len, mini_batch_size);
		CUDA_CHECK(cudaDeviceSynchronize());	

		// Copy gpu weights to device weights and compare for testing
		// CUDA_CHECK(cudaMemcpy(weights, weights_gpu, size_w, cudaMemcpyDeviceToHost));
		// for (int i = INPUT_SIZE * 30; i < weights_len; i++) {
		// 	std::cout <<  weights[i] << " ";
		// }
		// std::cout << std::endl;
		// delete[] dummy_w;

		CUDA_CHECK(cudaFree(nabla_w_gpu));
		CUDA_CHECK(cudaFree(nabla_b_gpu));	
		delete[] nabla_w;
		delete[] nabla_b;
	}

	std::tuple <float*, float*> backprop(int batch, int mini_batch_size, int *training_labels, float *training_data_gpu) {
		float *nabla_w_gpu, *nabla_b_gpu, *nabla_w_out, *nabla_b_out;
		float *activations_gpu, *zs_gpu, *y_gpu, *delta_gpu;

		int activations_len = std::accumulate(sizes.begin(), sizes.end(), 0) * mini_batch_size;
		int zs_len = std::accumulate(sizes.begin() + 1, sizes.end(), 0) * mini_batch_size;
		int size_activations = activations_len  * sizeof(float);
		int size_zs = zs_len * sizeof(float);

		// We want to make some ints of accumulated weights, biases and activations to make the next part easier
		int accum_w = 0, accum_b = 0, accum_z = 0, accum_a = 0;

		// Put nabla, activations, and zs on GPU
		CUDA_CHECK(cudaMalloc(&nabla_w_gpu, size_w * mini_batch_size));		
		CUDA_CHECK(cudaMalloc(&nabla_b_gpu, size_b * mini_batch_size));		
		CUDA_CHECK(cudaMalloc(&activations_gpu, size_activations));
		CUDA_CHECK(cudaMalloc(&zs_gpu, size_zs));
		// Set first TRAIN_SIZE activations to inputs
		CUDA_CHECK(cudaMemcpy(activations_gpu, &training_data_gpu[batch * INPUT_SIZE], INPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
		// block size = 1024
		dim3 block_size(32, 32);
		dim3 grid_size((sizes[1] + block_size.x - 1) / block_size.x, (mini_batch_size + block_size.y - 1) / block_size.y); 
		
		// FORWARD PASS
		for (int i = 0; i < num_layers - 1; i++) {
			if (i > 0) grid_size.x = (sizes[i + 1] + block_size.x - 1) / block_size.x;
			// Calculate z[i] = w[i] * a[i-1] + b[i]
			matmul_abT<<<grid_size, block_size>>>
			(&activations_gpu[accum_a], &weights_gpu[accum_w], &zs_gpu[accum_z], mini_batch_size, sizes[i + 1], sizes[i]);

			accum_a += sizes[i] * mini_batch_size; // updated accumulated A layers

			add_vec_to_mat<<<grid_size, block_size>>>
			(&zs_gpu[accum_z], &biases_gpu[accum_z], &zs_gpu[accum_z], mini_batch_size, sizes[i + 1]);

			// Calculate a[z[i]]
			sigmoid_vec<<<(sizes[i + 1] * mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(&zs_gpu[accum_a], &activations_gpu[accum_a], sizes[i + 1] * mini_batch_size);
			
			if (i < num_layers - 2) {
				accum_w += sizes[i] * sizes[i + 1];
				accum_b += sizes[i + 1];
				accum_z += sizes[i + 1] * mini_batch_size;
			}
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		// debuggin
		// float *zs = new float[zs_len];  
		// CUDA_CHECK(cudaMemcpy(zs, zs_gpu, size_zs, cudaMemcpyDeviceToHost));
		// std::cout << zs_len << std::endl;
		// std::cout << "Printing activations after forepass for batch sample " << batch << std::endl;
		// for (int k = 0; k < zs_len; k++) {
		// 	std::cout << zs[k] << " ";
		// }
		// std::cout << std::endl;
		// delete[] zs;

		// Vectorize our labels
		float *y = new float[OUTPUT_SIZE * mini_batch_size]();
		for (int i = 0; i < mini_batch_size; i++) {
			y[training_labels[batch + i] + i * mini_batch_size] = 1.0;
		}

		CUDA_CHECK(cudaMalloc(&y_gpu, OUTPUT_SIZE * mini_batch_size * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(y_gpu, y, OUTPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyHostToDevice));
		// BACKWARD PASS
		// Get delta
		CUDA_CHECK(cudaMalloc(&delta_gpu, size_b * mini_batch_size));
		grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
		grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;
		matrix_sub<<<grid_size, block_size>>>
		(&activations_gpu[accum_a], y_gpu, &delta_gpu[accum_z], mini_batch_size, OUTPUT_SIZE);

		// Debuggin
		// std::cout << "accum_a " << accum_a << " accum_z " << accum_z << " accum_w " << accum_w << " accum_b " << accum_b << std::endl; 
		// float *delta_db = new float[biases_len * mini_batch_size];  
		// CUDA_CHECK(cudaMemcpy(delta_db, delta_gpu, size_b * mini_batch_size, cudaMemcpyDeviceToHost));
		// std::cout << "Printing delta after backprop for batch sample " << batch << std::endl;
		// for (int i = sizes[1] * mini_batch_size; i < biases_len * mini_batch_size; i++) {
		// 	std::cout << delta_db[i] << " ";
		// }
		// std::cout << std::endl;
		// delete[] delta_db;

		// Add deltas to nabla_b
		CUDA_CHECK(cudaMemcpy(&nabla_b_gpu[accum_b * mini_batch_size], &delta_gpu[accum_b * mini_batch_size], OUTPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
		// Selecting dimensions naively for now
		dim3 block_size_3D(32, 16, 2);
		dim3 grid_size_3D((sizes[sizes.size() - 2] + block_size_3D.x - 1) / block_size_3D.x, 
		(OUTPUT_SIZE + block_size_3D.y - 1) / block_size_3D.y, 
		(mini_batch_size + block_size_3D.z - 1) / block_size_3D.z);
		
		accum_a -= sizes[sizes.size() - 2] * mini_batch_size;

		batch_outer_product<<<grid_size_3D, block_size_3D>>>
		(&delta_gpu[accum_z], &activations_gpu[accum_a], &nabla_w_gpu[accum_w], OUTPUT_SIZE, sizes[sizes.size() - 2], mini_batch_size, weights_len);
		// Backward loop
		for (int i = num_layers - 2; i > 0; i--) {
			accum_w -= sizes[i - 1] * sizes[i];
			accum_b -= sizes[i];
			accum_z -= sizes[i] * mini_batch_size;
			accum_a -= sizes[i - 1] * mini_batch_size;
			
			// Calculate sigmoid prime
			sigmoid_prime_vec<<<(sizes[i] * mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(&zs_gpu[accum_z], &zs_gpu[accum_z], sizes[i] * mini_batch_size);
			// Calculate Delta
			grid_size.x = (sizes[i] + block_size.x - 1) / block_size.x;
			matmul_ab<<<grid_size, block_size>>>
			(&delta_gpu[accum_z + sizes[i] * mini_batch_size], &weights_gpu[accum_w], &delta_gpu[accum_z], mini_batch_size, sizes[i+1], sizes[i]);

			grid_size.x = (sizes[i-1] + block_size.x - 1) / block_size.x;
			hadamard_mat<<<grid_size, block_size>>>
			(&delta_gpu[accum_z], &zs_gpu[accum_z], &delta_gpu[accum_z], mini_batch_size, sizes[i-1]);

			// Add deltas to nabla_b
			CUDA_CHECK(cudaMemcpy(&nabla_b_gpu[accum_b * mini_batch_size], &delta_gpu[accum_b * mini_batch_size], sizes[i] * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

			grid_size_3D.x = ((sizes[i - 1] + block_size_3D.x - 1) / block_size_3D.x);
			grid_size_3D.y = ((sizes[i] + block_size_3D.y - 1) / block_size_3D.y);
			batch_outer_product<<<grid_size_3D, block_size_3D>>>
			(&delta_gpu[accum_z], &activations_gpu[accum_a], &nabla_w_gpu[accum_w], sizes[i], sizes[i - 1], mini_batch_size, weights_len);
		}
		// CUDA_CHECK(cudaDeviceSynchronize());
		// Debuggin
		// float *activations = new float[activations_len];  
		// CUDA_CHECK(cudaMemcpy(activations, activations_gpu, size_activations, cudaMemcpyDeviceToHost));
		// std::cout << "Printing activations after backprop for batch sample " << batch << std::endl;
		// for (int i = INPUT_SIZE * mini_batch_size; i < activations_len; i++) {
		// 	std::cout << activations[i] << " ";
		// }
		// std::cout << std::endl;
		// delete[] activations;

		// Loop over nablas and return sums
		CUDA_CHECK(cudaMalloc(&nabla_w_out, size_w));
		CUDA_CHECK(cudaMalloc(&nabla_b_out, size_b));
		CUDA_CHECK(cudaMemset(nabla_w_out, 0, size_w));
		CUDA_CHECK(cudaMemset(nabla_b_out, 0, size_b));
		for (int i = 0; i < mini_batch_size; i++) {
			vector_add<<<(weights_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nabla_w_out, &nabla_w_gpu[i * weights_len], nabla_w_out, weights_len);
			vector_add<<<(biases_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nabla_b_out, &nabla_b_gpu[i * biases_len], nabla_b_out, biases_len);
		}
		// Debuggin
		// float *nabla_db = new float[weights_len];  
		// CUDA_CHECK(cudaMemcpy(nabla_db, nabla_w_out, size_w, cudaMemcpyDeviceToHost));
		// std::cout << "Printing weights after backprop for batch sample " << batch << std::endl;
		// for (int i = sizes[0] * sizes[1]; i < weights_len; i++) {
		// 	std::cout << nabla_db[i] / mini_batch_size << " ";
		// }
		// std::cout << std::endl;
		// delete[] nabla_db;

		// Free memory
		delete[] y;
		CUDA_CHECK(cudaFree(nabla_w_gpu));
		CUDA_CHECK(cudaFree(nabla_b_gpu));
		CUDA_CHECK(cudaFree(delta_gpu));
		CUDA_CHECK(cudaFree(y_gpu));
		CUDA_CHECK(cudaFree(zs_gpu));
		CUDA_CHECK(cudaFree(activations_gpu));

		return std::make_tuple(nabla_w_out, nabla_b_out);
	}
};
int main() {
 
    std::tuple<int*, float*, int*, float*> data_tuple; 
    std::cout << "Loading MNIST dataset" << std::endl;
    data_tuple = get_data_tuples();
    if (!std::get<0>(data_tuple)) {
        std::cout << "Empty dataset!" << std::endl;
    }
	
	std::vector<int> sizes = {INPUT_SIZE, 30, OUTPUT_SIZE};
	int mini_batch_size = 10;
	CUDA_NN nn(sizes);
	nn.SGD(std::get<0>(data_tuple), std::get<1>(data_tuple), EPOCHS, mini_batch_size, LEARNING_RATE, LAMBDA, false, std::get<2>(data_tuple), std::get<3>(data_tuple));

	// Free memory used by mnist data
	delete[] std::get<0>(data_tuple);
    delete[] std::get<1>(data_tuple);
	delete[] std::get<2>(data_tuple);
	delete[] std::get<3>(data_tuple);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
} 