// Header file for reading mnist data
#ifndef READ_MNIST
#define READ_MNIST

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

// Read mnist training data
std::vector<std::vector<double>> read_mnist(std::string a);

// function that will return tuple (training features, training labels, test features, test labels)
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_data_tuples();


// function that returns vectorized version of label
std::vector<double> vectorize_y(double y, size_t output_size);

#endif // READ_MNIST