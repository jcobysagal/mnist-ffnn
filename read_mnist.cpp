#include "read_mnist.h"
#include "mnist_macros.h"

// Read mnist training data
std::tuple<int*, float*> read_mnist(std::string a) {

	// Get path to data
	std::filesystem::path current_path = std::filesystem::current_path();

	std::filesystem::path parent_path = current_path.parent_path();

	std::filesystem::path parent_parent_path = parent_path.parent_path();

	std::ifstream file;
	int* labels;
    float *data;
	if (a == "test") {
		std::filesystem::path file_path = parent_parent_path / "data/mnist_dataset/mnist_test.csv";
		file.open(file_path);
		labels = (int*)malloc(TEST_SIZE * sizeof(int));
        data = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
	}
	else if (a == "train") {
		std::filesystem::path file_path = parent_parent_path / "data/mnist_dataset/mnist_train.csv";
		file.open(file_path);
		labels = (int*)malloc(TRAIN_SIZE * sizeof(int));
        data = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
	}
	else {
		std::cout << "Invalid input. Please enter 'test' or 'train'." << std::endl;
		return {};
	}
	if (file.is_open()) {
		std::string line;
        int data_i = 0; // counter for data allocation
		int labels_i = 0; // counter for labels allocation
		while (std::getline(file, line)) {
			std::istringstream ss(line);
			std::string token;
			bool label = true; // to normalize everything but the label
			while (std::getline(ss, token, ',')) {
				if (label) {
					labels[labels_i] = std::stoi(token);
					label = false;
                    labels_i++;
				}
				else {
					data[data_i] = std::stof(token) / 255.0;
                    data_i++;
				}
			}
		}
	}
    else {
        std::cout << "Error: Could not open file!" << std::endl;
        return {};
    }
	return std::make_tuple(labels, data);
}
// function that will return tuple (training features, training labels, test features, test labels)
std::tuple<int*, float*, int*, float*> get_data_tuples() {

	std::tuple<int*, float*> training_data;
	std::tuple<int*, float*> test_data;

	std::string a = "train";
	training_data = read_mnist(a);
	a = "test";
	test_data = read_mnist(a);

	return std::make_tuple(std::get<0>(training_data),std::get<1>(training_data),std::get<0>(test_data), std::get<1>(test_data));
}