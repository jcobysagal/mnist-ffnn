// Header file containing the functions to read in data for MNIST
#ifndef READ_MNIST
#define READ_MNIST

// Includes
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <tuple>

// Read mnist training data
std::tuple<int*, float*> read_mnist(std::string a);
// function that will return tuple (training features, training labels, test features, test labels)
std::tuple<int*, float*, int*, float*> get_data_tuples();

#endif