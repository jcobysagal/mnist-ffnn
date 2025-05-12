#include "utils.h"

// Seed the random number generator
static std::random_device rd;
static std::mt19937 gen(rd());
// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
// printvec function for debugging
void printvec(float *vec, int sz) {
	for (int i = 0; i < sz; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";
}
// printvec function for debugging
void printvec(int *vec, int sz) {
	for (int i = 0; i < sz; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";
}
// Create array of random numbers
void randnorm(float* arr, int sz) {
	// Define the "standard" normal distribution
	std::normal_distribution<> d(0, 1);
    for (int i = 0; i < sz; i++) {
	    arr[i] = d(gen);
    }
}
// Function to generate batch order for stochastic gradient descent 
std::vector<int> generateShuffledArray(int n, int mini_batch_size) {
    // Create a vector with numbers from 0 to n - 1
    std::vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        numbers[i] = i * mini_batch_size;
    }
    // Shuffle the vector using a random engine
    std::shuffle(numbers.begin(), numbers.end(), gen);
    return numbers;
}
// get max index of C array
int get_max_index(float* res) {
	int max_index = 0;

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (res[i] > res[max_index]) {
            max_index = i;
        }
	}
	return max_index;
}