#ifndef UTILS
#define UTILS

// Includes
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>

// Headers
#include <time.h>

#include "mnist_macros.h"

// Function to measure execution time
double get_time();
// printvec function for debugging
void printvec(float *vec, int sz);
// printvec function for debugging
void printvec(int *vec, int sz);
// Function to generate batch order for stochastic gradient descent 
std::vector<int> generateShuffledArray(int n, int mini_batch_size);
// random number generator from normal distribution
void randnorm(float *arr, int sz);
// get max index of C array
int get_max_index(float* res);

#endif