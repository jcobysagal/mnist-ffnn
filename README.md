# mnist-ffnn
MNIST classification with a Feed-Forward Neural Network. Includes both a CPU-powered model from scratch in C++ as well as a GPU-Accelerated model using CUDA, also from scratch.

You will need the csv files from here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

The C++ Model:
This is a model inspired by Michael Nielsen's code in his "Neural Networks and Deep Learning" Textbook. It is not the most optimal implementation but it was a great way for me to learn how to implement something like this completely from scratch in the language.

The CUDA Model:
I ported this model from my C++ code as a way to learn the ins and outs of CUDA and how to accelerate the training and inference times of my model. It is currently a straight port of my C++ model but I am currently working on vectorizing the mini-batches so that more things can be parallelized. You will need a CUDA compatible GPU to run this as well as the CUDA toolkit. I used CUDA toolkit v12.8 in building this model.

This was all implemented on my rig consisting of an AMD Ryzen 7 7800X3D CPU and NVIDIA RTX 4080 Super GPU. Results may vary.
