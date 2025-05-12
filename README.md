# mnist-ffnn
MNIST classification with a Feed-Forward Neural Network. Includes both a CPU-powered model from scratch in C++ as well as a GPU-Accelerated model using CUDA, also from scratch.

You will need the csv files from here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. Put these in a folder called "data" in the parent folder of the repo, or edit the read_mnist function to read from wherever you store it. 

## The C++ Model:
This is a model inspired by Michael Nielsen's code in his "Neural Networks and Deep Learning" Textbook. I could have used some libraries to optimize it but I wanted to learn how to implement something like this completely from scratch in the language.

## The CUDA Model:
I ported this model from my C++ code as a way to learn the ins and outs of CUDA and how to accelerate the training and inference times of my model. I have vectorized the model so that it trains over entire mini-batches in parallel and it achieves a good speedup over the super-naive, single-threaded C++ implementation.

I have also finally added a CMakeLists.txt! Just do the following to build the project:

From the repo root folder in terminal do:

```bash
$ mkdir build && cd build  
$ cmake ..  
$ cmake --build .  
$ ./mnist_ffnn
``` 

This was all implemented on my rig consisting of an AMD Ryzen 7 7800X3D CPU and NVIDIA RTX 4080 Super GPU. Results may vary.
