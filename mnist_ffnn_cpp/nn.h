#ifndef NN_H
#define NN_H

#include <vector>
#include <random>
#include "nn_linalg.h"

// Function to measure execution time
double get_time();

// sigmoid function
double sigmoid(double z);

// sigmoid prime function
double sigmoid_prime(double z);

// random number generator based on numpy.random.randn
std::vector<std::vector<double>> randn(int x, int y);

// shuffle vector for SGD
void shuffle_data(std::vector<std::vector<double>>& data);

class NeuralNetwork{
public:
	// Constructor
	NeuralNetwork(std::vector<int> sizes, bool large_weights = false);

	std::vector<double> feedforward(std::vector<double> a);

	void SGD(std::vector<std::vector<double>> training_data, int epochs, int mini_batch_size, float eta, float lmbda, bool val, std::vector<std::vector<double>> test_data = {});

private:
	//attributes
	std::vector<int> sizes;
	size_t num_layers;
	std::vector<std::vector<std::vector<double>>> biases;
	std::vector<std::vector<std::vector<double>>> weights;

	//function to evaluate performance of the neural network 
	int evaluate(std::vector<std::vector<double>>& test_data);

	// Update the network's weights and biases by applying
	// gradient descent using backpropagation to a single mini batch.
	void update_mini_batch(std::vector<std::vector<double>>& mini_batch, float eta, float lmbda, int training_data_size);

	// Backpropogation
	std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> backprop(std::vector<double> x, std::vector<double> y);

};
#endif // NN_H