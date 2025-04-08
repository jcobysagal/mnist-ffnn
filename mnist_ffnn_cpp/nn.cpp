#include <iostream>
#include <tuple>
#include <algorithm>
#include <time.h>

#include "nn.h"
#include "nn_linalg.h"
#include "read_mnist.h"

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// random number generator based on numpy.random.randn
std::vector<std::vector<double>> randn(int x, int y) {
	// Seed the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Define the "standard" normal distribution
	std::normal_distribution<> d(0, 1);

	// Generate the vector
	std::vector<std::vector<double>> vec;
	for (int i = 0; i < y; i++) {
		std::vector<double> temp;
		for (int j = 0; j < x; j++) {
			temp.push_back(d(gen));
		}
		vec.push_back(temp);
	}
	return vec;
}

// shuffle vector for SGD
void shuffle_data(std::vector<std::vector<double>>& data) {
	std::random_device rd;
	std::mt19937 generator(rd());

	// Shuffle the vector
	std::shuffle(data.begin(), data.end(), generator);
}

/*
Class for feed forward neural network
*/

// Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> sizes, bool large_weights) {
	std::cout << "Initializing network..." << std::endl;

	num_layers = sizes.size();
	this->sizes = sizes;

	for (auto it = sizes.begin() + 1; it != sizes.end(); ++it) {
		biases.push_back(randn(*it, 1));
		weights.push_back(randn(*std::prev(it, 1), *it));
	}

	if (!large_weights) {
		for (int i = 0; i < weights.size(); i++) {
			for (int j = 0; j < weights[i].size(); j++) {
				for (int k = 0; k < weights[i][j].size(); k++) {
					weights[i][j][k] /= pow(sizes[i], 0.5);
				}
			}
		}
	}
	std::cout << "Initialization complete" << std::endl;

	std::cout << "Dimensions of weights" << std::endl;
	for (int i = 0; i < weights.size(); i++) {
		std::cout << "(" << weights[i].size() << "," << weights[i][0].size() << ")" << " ";
	}
	std::cout << std::endl;
	std::cout << "Dimensions of biases" << std::endl;
	for (int i = 0; i < biases.size(); i++) {
		std::cout << biases[i][0].size() << " ";
	}
	std::cout << std::endl;
}

std::vector<double> NeuralNetwork::feedforward(std::vector<double> a) {
	// Return the output of the network if ``a`` is input
	for (int layer = 0; layer < num_layers - 1; layer++) {
		// calculate z
		std::vector<double> z;
		std::vector<double> temp;

		z = add_vec(mat_vec(weights[layer], a), biases[layer][0]);
		for (int i = 0; i < sizes[layer + 1]; i++) {
			temp.push_back(sigmoid(z[i]));
		}
		a = temp;
	}
	return a;
}

//function to evaluate performance of the neural network 
int NeuralNetwork::evaluate(std::vector<std::vector<double>>& test_data) {
	// Separate test_data into data and labels
	std::vector<std::vector<double>> x;
	std::vector<double> y;
	int successes = 0;

	for (std::vector<double>& data : test_data) {
		y.push_back(data[0]);
		x.push_back(std::vector<double>(data.begin() + 1, data.end()));
	}

	for (int i = 0; i < test_data.size(); i++) {
		std::vector<double> res = feedforward(x[i]);
		auto it = std::max_element(res.begin(), res.end());
		if (std::distance(res.begin(), it) == y[i]) {
			successes++;
		}
	}

	return successes;
}

void NeuralNetwork::SGD(std::vector<std::vector<double>> training_data, int epochs, int mini_batch_size, float eta, float lmbda, bool val, std::vector<std::vector<double>> test_data) {
	//	Train the neural network using mini-batch stochastic
	//	gradient descent.The "training_data" is a list of tuples
	//	"(x, y)" representing the training inputs and the desired
	//	outputs.The other non - optional parameters are
	//	self - explanatory.If "test_data" is provided then the
	//	network will be evaluated against the test data after each
	//	epoch, and partial progress printed out.This is useful for
	//	tracking progress, but slows things down substantially.
	std::cout << "Beginning stochastic gradient descent" << std::endl;
	std::vector<std::vector<double>> val_data;

	if (test_data.size() > 0) {
		int n_test_data = test_data.size();
	}
	if (val) {
		int n_val_data = 10000;
		
		val_data = std::vector<std::vector<double>>(training_data.begin() + training_data.size() - n_val_data, training_data.end());
		training_data  = std::vector<std::vector<double>>(training_data.begin(), training_data.end() - n_val_data);
	}
	int num_batches = training_data.size() / mini_batch_size;
	std::vector<std::vector<double>> mini_batch;
	for (int epoch = 1; epoch <= epochs; epoch++) {
		double st = get_time();
		shuffle_data(training_data);
		// make mini-batches
		for (int batch = 0; batch < num_batches; batch++) {
			mini_batch = std::vector<std::vector<double>>(training_data.begin() + batch * mini_batch_size, training_data.begin() + batch * mini_batch_size + mini_batch_size);
			update_mini_batch(mini_batch, eta, lmbda, training_data.size());
		}
		if (test_data.size() > 0) {
			std::cout << "Test data accuracy: " << evaluate(test_data) << "/" << test_data.size() << " correct." << std::endl;
		}
		if (val) {
			std::cout << "Val data accuracy: " << evaluate(val_data) << "/" << val_data.size() << " correct." << std::endl;
		}
		double ft = get_time() - st; 
		std::cout << "Epoch " << epoch << " complete." << std::endl;
		std::cout << "Time elapse = " << ft << "seconds" << std::endl;
	}
}

//	"""Update the network's weights and biases by applying
//	gradient descent using backpropagation to a single mini batch.
void NeuralNetwork::update_mini_batch(std::vector<std::vector<double>>& mini_batch, float eta, float lmbda, int training_data_size) {
	std::vector<std::vector<std::vector<double>>> nabla_b;
	std::vector<std::vector<std::vector<double>>> nabla_w;

	// make nabla zeros
	for (auto it = sizes.begin() + 1; it != sizes.end(); ++it) {
		std::vector<std::vector<double>> temp_b;
		std::vector<double> temp(*it, 0.0);
		// populate temp_b with zeros
		temp_b.push_back(temp);
		// populate temp_w with zeros
		nabla_b.push_back(temp_b);
	}
	for (auto it = sizes.begin() + 1; it != sizes.end(); ++it) {
		std::vector<double> temp(*std::prev(it, 1), 0.0);
		std::vector<std::vector<double>> temp_w(*it, temp);
		nabla_w.push_back(temp_w);
	}

	// Separate mini_batch into data and labels
	std::vector<std::vector<double>> x;
	std::vector<std::vector<double>> y;

	for (std::vector<double>& data : mini_batch) {
		y.push_back(vectorize_y(data[0], sizes[sizes.size() - 1]));
		x.push_back(std::vector<double>(data.begin() + 1, data.end()));
	}

	//backprop
	for (int i = 0; i < mini_batch.size(); i++) {
		std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> delta_nabla = backprop(x[i], y[i]);
		for (int p = 0; p < nabla_w.size(); p++) {
			for (int j = 0; j < nabla_w[p].size(); j++) {
				for (int k = 0; k < nabla_w[p][j].size(); k++) {
					nabla_w[p][j][k] = nabla_w[p][j][k] + std::get<0>(delta_nabla)[p][j][k];
				}
			}
		}
		for (int p = 0; p < nabla_b.size(); p++) {
			for (int j = 0; j < nabla_b[p].size(); j++) {
				for (int k = 0; k < nabla_b[p][j].size(); k++) {
					nabla_b[p][j][k] = nabla_b[p][j][k] + std::get<1>(delta_nabla)[p][j][k];
				}
			}
		}
	}

	// update weights and biases
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				weights[i][j][k] = (1 - eta * (lmbda / training_data_size)) * weights[i][j][k] - (eta / mini_batch.size()) * nabla_w[i][j][k];
			}
		}
	}
	for (int i = 0; i < biases.size(); i++) {
		for (int j = 0; j < biases[i].size(); j++) {
			for (int k = 0; k < biases[i][j].size(); k++) {
				biases[i][j][k] = biases[i][j][k] - (eta / mini_batch.size()) * nabla_b[i][j][k];
			}
		}
	}
}

// Backpropogation
std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> NeuralNetwork::backprop(std::vector<double> x, std::vector<double> y) {
	std::vector<std::vector<std::vector<double>>> nabla_b;
	std::vector<std::vector<std::vector<double>>> nabla_w;
	std::vector<double> activation;
	std::vector<std::vector<double>> activations;
	std::vector<double> z;
	std::vector<std::vector<double>> zs;
	std::vector<double> delta;
	std::vector<double> sig_prime;

	// make nabla zeros
	for (auto it = sizes.begin() + 1; it != sizes.end(); ++it) {
		std::vector<std::vector<double>> temp_b;
		std::vector<double> temp(*it, 0.0);
		// populate temp_b with zeros
		temp_b.push_back(temp);
		// populate temp_w with zeros
		nabla_b.push_back(temp_b);
	}
	for (auto it = sizes.begin() + 1; it != sizes.end(); ++it) {
		std::vector<double> temp(*std::prev(it, 1), 0.0);
		std::vector<std::vector<double>> temp_w(*it, temp);
		nabla_w.push_back(temp_w);
	}

	activation = x;
	activations.push_back(activation);

	// forward pass
	for (int i = 0; i < num_layers - 1; i++) {
		z = add_vec(mat_vec(weights[i], activation), biases[i][0]);
		zs.push_back(z);
		std::vector<double> temp_a;
		for (double& val : z) {
			temp_a.push_back(sigmoid(val));
		}
		activation = temp_a;
		temp_a.clear();
		activations.push_back(activation);
	}

	//delta = quadratic_cost_derivative(activations[activations.size() - 1], y, zs[zs.size() - 1]);
	delta = cross_entropy_cost_derivative(activations[activations.size() - 1], y);
	nabla_b[nabla_b.size() - 1][0] = delta;
	nabla_w[nabla_w.size() - 1] = outer_product(delta, activations[activations.size() - 2]);

	// backward loop
	for (int layer = num_layers - 2; layer > 0; layer--) {
		z = zs[layer - 1];

		sig_prime.clear();
		for (double& val : z) {
			sig_prime.push_back(sigmoid_prime(val));
		}
		// calculate delta
		delta = hadamard_product(mat_vec(transpose_matrix(weights[layer]), delta), sig_prime);
		nabla_b[layer - 1][0] = delta;
		nabla_w[layer - 1] = outer_product(delta, activations[layer - 1]);
	}

	return std::make_tuple(nabla_w, nabla_b);
}