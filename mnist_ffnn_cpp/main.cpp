/*
Runs training protocol for Feed Forward Neural Net based on MNIST dataset
Should be able to take other datasets as well
Ported from Michael Nielsen's Neural Networks and Deep Learning Python Code
Takes 4 parameters:
sizes - vector<int> containing shape of neural network. {input, hidden layers, output}
epochs - int number of training epochs
mini_batch_size - int
eta - float learning rate
*/
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <string>
#include <random>

// Headers
#include "nn.h"
#include "read_mnist.h"

// main
int main() {

	std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data_tuple;
	int epochs = 30;
	int mini_batch_size = 10;
	float eta = 3.0;
	float lmbda = 0;
	bool val = false;
	
	std::cout << "Loading mnist dataset..." << std::endl;
	data_tuple = get_data_tuples();
	//std::vector<std::vector<double>> data = read_mnist("test"); // just load test data for debugging - loads faster
	std::vector<int> sizes = { 784, 30, 10 };
	NeuralNetwork nn(sizes);
	//nn.SGD(data, epochs, mini_batch_size, eta); // train NN with test data
	nn.SGD(std::get<0>(data_tuple), epochs, mini_batch_size, eta, lmbda, val, std::get<1>(data_tuple));

	return 0;
}
