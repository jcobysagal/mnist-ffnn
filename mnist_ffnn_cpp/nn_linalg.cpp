#include "nn_linalg.h"

#include <iostream>

// matrix transpose
std::vector<std::vector<double>> transpose_matrix(const std::vector<std::vector<double>>& matrix) {
	int rows = matrix.size();
	int cols = matrix[0].size();
	std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			transposedMatrix[j][i] = matrix[i][j];
		}
	}
	return transposedMatrix;
}
// add two vectors
std::vector<double> add_vec(std::vector<double> a, std::vector<double> b) {
	std::vector<double> c;

	if (a.size() != b.size()) {
		std::cout << "Cannot add vectors. a has dims " << a.size() << " and b has dims " << b.size() << std::endl;
		return {};
	}

	for (int i = 0; i < a.size(); i++) {
		c.push_back(a[i] + b[i]);
	}
	return c;
}

// subtract two vectors
std::vector<double> sub_vec(std::vector<double> a, std::vector<double> b) {
	std::vector<double> c;

	if (a.size() != b.size()) {
		std::cout << "Cannot subtract vectors. a has dims " << a.size() << " and b has dims " << b.size() << std::endl;
		return {};
	}

	for (int i = 0; i < a.size(); i++) {
		c.push_back(a[i] - b[i]);
	}
	return c;
}

// dot product of two vectors
double dot(std::vector<double> a, std::vector<double> b) {

	if (a.size() != b.size()) {
		std::cout << "Cannot multiply vectors. a has dims " << a.size() << " and b has dims " << b.size() << std::endl;
		return {};
	}

	double c = 0;
	for (int i = 0; i < a.size(); i++) {
		c += a[i] * b[i];
	}
	return c;
}

// outer product of two vectors
std::vector<std::vector<double>> outer_product(const std::vector<double>& u, const std::vector<double>& v) {
	int m = u.size();
	int n = v.size();
	std::vector<std::vector<double>> result(m, std::vector<double>(n));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			result[i][j] = u[i] * v[j];
		}
	}
	return result;
}

// hadamard product of two vectors
std::vector<double> hadamard_product(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	if (vec1.size() != vec2.size()) {
		throw std::invalid_argument("Vectors must have the same size");
	}

	std::vector<double> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}

// hadamard product of two matrices
std::vector<std::vector<int>> hadamard_product(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2) {
	if (matrix1.empty() || matrix2.empty() || matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
		throw std::invalid_argument("Matrices must have the same dimensions and be non-empty.");
	}

	int rows = matrix1.size();
	int cols = matrix1[0].size();
	std::vector<std::vector<int>> result(rows, std::vector<int>(cols));

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = matrix1[i][j] * matrix2[i][j];
		}
	}

	return result;
}

// matrix-vector multiplication
std::vector<double> mat_vec(std::vector<std::vector<double>> a, std::vector<double> b) {

	if (a[0].size() != b.size()) {
		std::cout << "Cannot multiply matrix and vector. a[0] has dims " << a[0].size() << " and b has dims " << b.size() << std::endl;
		return {};
	}

	std::vector<double> c;
	for (int i = 0; i < a.size(); i++) {
		c.push_back(dot(a[i], b));
	}
	return c;
}

// matrix-matrix multiplication - might not use this, but will be good to have
std::vector<std::vector<double>> mat_mat(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {
	std::vector<std::vector<double>> c;
	for (int i = 0; i < a.size(); i++) {
		std::vector<double> temp;
		for (int j = 0; j < b[0].size(); j++) {
			std::vector<double> temp2;
			for (int k = 0; k < b.size(); k++) {
				temp2.push_back(a[i][k] * b[k][j]);
			}
			temp.push_back(dot(a[i], temp2));
		}
		c.push_back(temp);
	}
	return c;
}

// sigmoid function
double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

// sigmoid prime function
double sigmoid_prime(double z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

std::vector<double> quadratic_cost_derivative(std::vector<double> outputs, std::vector<double> labels, std::vector<double> z_fin) {
	std::vector<double> sig_prime;

	for (double& val : z_fin) {
		sig_prime.push_back(sigmoid_prime(val));
	}
	return hadamard_product(sub_vec(outputs, labels), sig_prime);
}

std::vector<double> cross_entropy_cost_derivative(std::vector<double> outputs, std::vector<double> labels) {
	return sub_vec(outputs, labels);
}