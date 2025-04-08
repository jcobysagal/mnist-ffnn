#include "read_mnist.h"

#include <iostream>
#include <vector>

// Read mnist training data
std::vector<std::vector<double>> read_mnist(std::string a) {
	std::ifstream file;
	if (a == "test") {
		file.open("mnist_test.csv");
	}
	else if (a == "train") {
		file.open("mnist_train.csv");
	}
	else {
		std::cout << "Invalid input. Please enter 'test' or 'train'." << std::endl;
		return {};
	}
	std::vector<std::vector<double>> data;
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			std::vector<double> row;
			std::istringstream ss(line);
			std::string token;
			int toke = 0; // to normalize everything but the label
			while (std::getline(ss, token, ',')) {
				if (toke == 0) {
					row.push_back(std::stod(token));
					toke++;
				}
				else {
					row.push_back(std::stod(token) / 255.0);
				}
			}
			data.push_back(row);
		}
	}
	return data;
}

// function that will return tuple (training features, training labels, test features, test labels)
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_data_tuples() {

	std::vector<std::vector<double>> training_data;
	std::vector<std::vector<double>> test_data;

	std::string a = "train";
	training_data = read_mnist(a);
	a = "test";
	test_data = read_mnist(a);

	return std::make_tuple(training_data, test_data);
	
}

// function that returns vectorized version of label
std::vector<double> vectorize_y(double y, size_t output_size) {
	std::vector<double> vectorized_y(output_size, 0.0);
	vectorized_y[y] = 1.0;
	return vectorized_y;
}