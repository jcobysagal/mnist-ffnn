/*
struct for neural network layer
I got tired of all of the nested arrays that my code ended up with, so here is a new approach
Simple struct for each layer containing all of the variables for that layer
*/
#ifndef NN_LAYER
#define NN_LAYER

#include "mnist_macros.h"

typedef struct {

    int num_neurons;
    float *weights;
    float *biases;

    float *d_weights;
    float *d_biases;
    float *nabla_w;
    float *nabla_b;
    float *zs;
    float *activations;    
    float *deltas;
    // output nablas for backpropagation
    float *nabla_w_out;
    float *nabla_b_out;

} layer;

#endif
