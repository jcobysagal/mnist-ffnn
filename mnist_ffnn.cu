/*
Third rework of my CUDA-Accelerated mnist Feed-Forward Neural Network!
Second attempt definitely was an improvement over my first attempt, but I figured out a much better way to organize the neural net with structs
*/

//Includes
#include <iostream>
#include <tuple>
#include <vector>
#include <random>

// Headers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "read_mnist.h"
#include "mnist_macros.h"
#include "utils.h"
#include "nn_kernels.h"
#include "nn_layer.h"

class CUDA_NN{
	public:
	// Constructor
	CUDA_NN(int *sizes,  int num_layers, int mini_batch_size, bool large_weights = false) {
		std::cout << "Initializing network..." << std::endl;
		this->num_layers = num_layers;
		this->sizes = sizes;
		this->large_weights = large_weights;
		this->mini_batch_size = mini_batch_size;
		
		// Initialize network
		nn = (layer**)malloc(num_layers * sizeof(layer*));
		// Initalize arrays of weights and biases which will contain pointers to the GPU
		for (int i = 0; i < num_layers; i++) {
			nn[i] = (layer*)malloc(sizeof(layer));
			nn[i]->num_neurons = sizes[i];
			
			if (i > 0) {
				int weights_dims = nn[i - 1]->num_neurons * nn[i]->num_neurons;
				int biases_dims = nn[i]->num_neurons;

				nn[i]->weights = (float*)malloc(weights_dims * sizeof(float));
				nn[i]->biases = (float*)malloc(biases_dims * sizeof(float));

				initialize_weights(nn[i], weights_dims, large_weights);
				initialize_biases(nn[i], biases_dims, sizes);
				
				CUDA_CHECK(cudaMalloc(&nn[i]->d_weights, weights_dims * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->d_biases, biases_dims * sizeof(float)));

				CUDA_CHECK(cudaMemcpy(nn[i]->d_weights, nn[i]->weights, weights_dims * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(nn[i]->d_biases, nn[i]->biases, biases_dims * sizeof(float), cudaMemcpyHostToDevice));

				CUDA_CHECK(cudaMalloc(&nn[i]->nabla_w, weights_dims * mini_batch_size * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->nabla_b, biases_dims * mini_batch_size * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->nabla_w_out, weights_dims * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->nabla_b_out, biases_dims * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->zs, nn[i]->num_neurons * mini_batch_size * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->activations, nn[i]->num_neurons * mini_batch_size * sizeof(float)));
				CUDA_CHECK(cudaMalloc(&nn[i]->deltas, nn[i]->num_neurons * mini_batch_size * sizeof(float)));
			}
		}
		std::cout << "Initialization complete\n";
	}
	void SGD(int *training_labels, float *training_data, int epochs, float eta, float lmbda, bool val, int *test_labels, float *test_data){
		std::cout << "Beginning stochastic gradient descent" << std::endl;

		float *training_data_gpu, *test_data_gpu;
		int train_size = TRAIN_SIZE * INPUT_SIZE * sizeof(float);
		int test_size = TEST_SIZE * INPUT_SIZE * sizeof(float);
		// check null pointers
		if (training_data == NULL) {
			printf("WARNING: TRAINING DATA IS NULL");
			return;
		}
		// Copy data to GPU
		CUDA_CHECK(cudaMalloc(&training_data_gpu, train_size));
		CUDA_CHECK(cudaMemcpy(training_data_gpu, training_data, train_size, cudaMemcpyHostToDevice));
		if (test_data != NULL) {
			CUDA_CHECK(cudaMalloc(&test_data_gpu, test_size));
			CUDA_CHECK(cudaMemcpy(test_data_gpu, test_data, test_size, cudaMemcpyHostToDevice));
		}
		int num_batches = TRAIN_SIZE / mini_batch_size;
		for (int epoch = 1; epoch <= EPOCHS; epoch++) {

			std::cout << "Beginning epoch " << epoch << std::endl;
			double st = get_time();

			batch_order = generateShuffledArray(num_batches, mini_batch_size);
			for (int i = 0; i < num_batches; i++) {
				update_mini_batch(batch_order[i], training_labels, training_data_gpu, LEARNING_RATE, LAMBDA, TRAIN_SIZE);
			}
			if (test_data != NULL) {
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
		for (int i = 0; i < num_layers; i++){

			if (i > 0) {
				// Device Memory
				CUDA_CHECK(cudaFree(nn[i]->d_weights));
				CUDA_CHECK(cudaFree(nn[i]->d_biases));
				CUDA_CHECK(cudaFree(nn[i]->nabla_w));
				CUDA_CHECK(cudaFree(nn[i]->nabla_w_out));
				CUDA_CHECK(cudaFree(nn[i]->nabla_b));
				CUDA_CHECK(cudaFree(nn[i]->nabla_b_out));
				CUDA_CHECK(cudaFree(nn[i]->zs));
				CUDA_CHECK(cudaFree(nn[i]->activations));
				CUDA_CHECK(cudaFree(nn[i]->deltas));
				// Host memory
				free(nn[i]->weights);
				free(nn[i]->biases);
			}

			// Layer memeory
			free(nn[i]);
		}
		free(nn);
	}
	private:
	layer **nn;
	int *sizes;
	int mini_batch_size;
	std::vector<int> batch_order;
	int num_layers;
	bool large_weights;

	void initialize_weights(layer *layer, int weights_dims, bool large_weights) {
		// Populate arrray with random values
		randnorm(layer->weights, weights_dims);
		// Rescale option
		if (!large_weights) {
			for (int i = 0; i < weights_dims; i++) {	
					layer->weights[i] /= pow(weights_dims / layer->num_neurons, 0.5); // weights_dims/num_neurons gets us prev layer->num_neurons
				}
		}
	}
	void initialize_biases(layer *layer, int biases_dims, int *sizes) {
		randnorm(layer->biases, biases_dims);
	}
	// Feedforward function
	float* feedforward(float *input_gpu) {
		// set input layer activations to point to input
		nn[0]->activations = input_gpu;
		// Put zs and activations on GPU
		for (int i = 1; i < num_layers; i++){
			CUDA_CHECK(cudaMemset(nn[i]->zs, 0, nn[i]->num_neurons *  sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->activations, 0, nn[i]->num_neurons * sizeof(float)));
		}
		
		dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid_size;
		// FORWARD PASS
		for (int i = 1; i < num_layers; i++) {
			grid_size.x = (nn[i]->num_neurons + block_size.x - 1) / block_size.x;
			grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;
			forward<<<grid_size, block_size>>>
			(nn[i-1]->activations, nn[i]->d_weights, nn[i]->d_biases, nn[i]->zs, nn[i]->activations, nn[i-1]->num_neurons, nn[i]->num_neurons, 1);
		}	
		CUDA_CHECK(cudaDeviceSynchronize());
		// Copy result to output
		float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
		CUDA_CHECK(cudaMemcpy(output, nn[num_layers - 1]->activations, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

		return output;	
	}
	// function to evaluate performance of the neural network 
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
			// std::cout << "Prediction: " << m << ", Label: " << test_labels[i] << std::endl;
			if (m == test_labels[i]) {
				successes++; 
			}
			free(res);
		}
		return successes;
	}
	void update_mini_batch(int batch, int *training_labels, float *training_data, float eta, float lambda, int training_data_size) {
		// backpropagation
		backprop(batch, training_labels, training_data);

		// Update weights and biases
		for (int i = 1; i < num_layers; i++) {
			update_weights<<<(nn[i - 1]->num_neurons * nn[i]->num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(nn[i]->d_weights, nn[i]->nabla_w_out, nn[i]->d_weights, nn[i - 1]->num_neurons * nn[i]->num_neurons, mini_batch_size);
			update_biases<<<(nn[i]->num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(nn[i]->d_biases, nn[i]->nabla_b_out, nn[i]->d_biases, nn[i]->num_neurons, mini_batch_size);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	void backprop(int batch, int *training_labels, float *training_data_gpu) {

		for (int i = 1; i < num_layers; i++) {
			// Zero everything needed for GD
			CUDA_CHECK(cudaMemset(nn[i]->nabla_w, 0, nn[i - 1]->num_neurons * nn[i]->num_neurons * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->nabla_b, 0 , nn[i]->num_neurons  * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->zs, 0 , nn[i]->num_neurons  * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->activations, 0 , nn[i]->num_neurons  * mini_batch_size * sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->deltas, 0 , nn[i]->num_neurons  * mini_batch_size * sizeof(float)));
		}

		dim3 block_size;
		dim3 grid_size;
		
		// Set input layer activations
		nn[0]->activations = &training_data_gpu[batch * INPUT_SIZE]; 

		// FORWARD PASS
		for (int i = 1; i < num_layers; i++) {
			block_size.x = BLOCK_SIZE;
			block_size.y = BLOCK_SIZE;
			grid_size.x = (nn[i]->num_neurons + block_size.x - 1) / block_size.x;
			grid_size.y = (mini_batch_size + block_size.y - 1) / block_size.y;
			
			forward<<<grid_size, block_size>>>
			(nn[i - 1]->activations, nn[i]->d_weights, nn[i]->d_biases, nn[i]->zs, nn[i]->activations, nn[i - 1]->num_neurons, nn[i]->num_neurons, mini_batch_size);
			CUDA_CHECK(cudaGetLastError());
		}

		// Vectorize our labels
		float *ys_gpu;
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
		(nn[num_layers - 1]->activations, ys_gpu, nn[num_layers - 1]->deltas, mini_batch_size, OUTPUT_SIZE);
		CUDA_CHECK(cudaGetLastError());

		// Add deltas to nabla_b
		CUDA_CHECK(cudaMemcpy(nn[num_layers - 1]->nabla_b, nn[num_layers - 1]->deltas, OUTPUT_SIZE * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

		// 3D block
		dim3 block_size_3D(32, 16, 2);
		dim3 grid_size_3D((nn[num_layers - 2]->num_neurons + block_size_3D.x - 1) / block_size_3D.x, 
		(OUTPUT_SIZE + block_size_3D.y - 1) / block_size_3D.y, 
		(mini_batch_size + block_size_3D.z - 1) / block_size_3D.z);

		// Outer product to get nabla_w
		batch_outer_product<<<grid_size_3D, block_size_3D>>>
		(nn[num_layers - 1]->deltas, nn[num_layers - 2]->activations, nn[num_layers - 1]->nabla_w, OUTPUT_SIZE, nn[num_layers - 2]->num_neurons, mini_batch_size);
		CUDA_CHECK(cudaGetLastError());
		// BACKWARD PASS
		for (int i = num_layers - 2; i > 0; i--) {
			// Calculate sigmoid prime
			sigmoid_prime_vec<<<(nn[i]->num_neurons * mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
			(nn[i]->zs, nn[i]->zs, nn[i]->num_neurons * mini_batch_size);
			CUDA_CHECK(cudaGetLastError());

			grid_size.x = (nn[i]->num_neurons + block_size.x - 1) / block_size.x;
			matmul_ab<<<grid_size, block_size>>>
			(nn[i + 1]->deltas, nn[i + 1]->d_weights, nn[i]->deltas, mini_batch_size, nn[i + 1]->num_neurons, nn[i]->num_neurons);
			CUDA_CHECK(cudaGetLastError());

			grid_size.x = (sizes[i] + block_size.x - 1) / block_size.x;
			hadamard_mat<<<grid_size, block_size>>>
			(nn[i]->deltas, nn[i]->zs, nn[i]->deltas, mini_batch_size, nn[i]->num_neurons);
			CUDA_CHECK(cudaGetLastError());

			// Add deltas to nabla_b
			CUDA_CHECK(cudaMemcpy(nn[i]->nabla_b, nn[i]->deltas, nn[i]->num_neurons * mini_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

			grid_size_3D.x = ((nn[i - 1]->num_neurons + block_size_3D.x - 1) / block_size_3D.x);
			grid_size_3D.y = ((nn[i]->num_neurons + block_size_3D.y - 1) / block_size_3D.y);
			batch_outer_product<<<grid_size_3D, block_size_3D>>>
			(nn[i]->deltas, nn[i - 1]->activations, nn[i]->nabla_w, nn[i]->num_neurons, nn[i - 1]->num_neurons, mini_batch_size);	
			CUDA_CHECK(cudaGetLastError());			
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		// Loop over nablas and return sums
		for (int i = 1; i < num_layers; i++) {
			CUDA_CHECK(cudaMemset(nn[i]->nabla_w_out, 0, nn[i - 1]->num_neurons * nn[i]->num_neurons * sizeof(float)));
			CUDA_CHECK(cudaMemset(nn[i]->nabla_b_out, 0, nn[i]->num_neurons * sizeof(float)));
			for (int j = 0; j < mini_batch_size; j++) {
				vector_add<<<(nn[i - 1]->num_neurons * nn[i]->num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
				(nn[i]->nabla_w_out, &nn[i]->nabla_w[j * nn[i - 1]->num_neurons * nn[i]->num_neurons], nn[i]->nabla_w_out, nn[i - 1]->num_neurons * nn[i]->num_neurons);
				vector_add<<<(nn[i]->num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
				(nn[i]->nabla_b_out, &nn[i]->nabla_b[j * nn[i]->num_neurons], nn[i]->nabla_b_out, nn[i]->num_neurons);
			}
			CUDA_CHECK(cudaGetLastError());
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

		// Free ys
		free(ys);
		CUDA_CHECK(cudaFree(ys_gpu));
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
	int mini_batch_size = 32;

	CUDA_NN nn(sizes, num_layers, mini_batch_size);
	nn.SGD(std::get<0>(data_tuple), std::get<1>(data_tuple), EPOCHS, LEARNING_RATE, LAMBDA, false, std::get<2>(data_tuple), std::get<3>(data_tuple));

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