// Header file contaning the functions for the neural network
#ifndef NN_LINALG
#define NN_LINALG

#include <vector> // for data
#include <cmath> // for exp

// matrix transpose
std::vector<std::vector<double>> transpose_matrix(const std::vector<std::vector<double>>& matrix);
// add two vectors
std::vector<double> add_vec(std::vector<double> a, std::vector<double> b);

// subtract two vectors
std::vector<double> sub_vec(std::vector<double> a, std::vector<double> b);

// dot product of two vectors
double dot(std::vector<double> a, std::vector<double> b);

// outer product of two vectors
std::vector<std::vector<double>> outer_product(const std::vector<double>& u, const std::vector<double>& v);

// hadamard product of two vectors
std::vector<double> hadamard_product(const std::vector<double>& vec1, const std::vector<double>& vec2);

// hadamard product of two matrices
std::vector<std::vector<int>> hadamard_product(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2);

// matrix-vector multiplication
std::vector<double> mat_vec(std::vector<std::vector<double>> a, std::vector<double> b);

// matrix-matrix multiplication - might not use this, but will be good to have
std::vector<std::vector<double>> mat_mat(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b);

// sigmoid function
double sigmoid(double z);

// sigmoid prime function
double sigmoid_prime(double z);

// quadratic cost derivative
std::vector<double> quadratic_cost_derivative(std::vector<double> outputs, std::vector<double> labels, std::vector<double> z_fin);

// cross entropy cost derivative
std::vector<double> cross_entropy_cost_derivative(std::vector<double> outputs, std::vector<double> labels);
#endif // NN_LINALG