#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct Module Module;
typedef struct Sequential Sequential;

struct Module {
    Tensor* (*forward)(struct Module*, Tensor*);
    void (*backward)(struct Module*);
    void (*update)(struct Module*, float lr);
    void (*to)(struct Module*, Device device);
    void (*free)(struct Module*);
    Tensor** parameters;
    Tensor** gradients;
    size_t num_parameters;
};

// Linear (Fully Connected) Layer
Module* nn_linear(size_t in_features, size_t out_features);

// Convolutional Layer
Module* nn_conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding);

// Max Pooling Layer
Module* nn_maxpool2d(size_t kernel_size, size_t stride);

// Batch Normalization Layer
Module* nn_batchnorm2d(size_t num_features);

// Dropout Layer
Module* nn_dropout(float p);

// Activation functions
Module* nn_relu();
Module* nn_sigmoid();
Module* nn_tanh();

// Sequential model (a stack of layers)
Sequential* nn_sequential(Module** modules, size_t num_modules);

// Forward pass through a sequential model
Tensor* sequential_forward(Sequential* model, Tensor* input);

// Backward pass through a sequential model
void sequential_backward(Sequential* model);

// Update parameters of a sequential model
void sequential_update(Sequential* model, float lr);

// Move sequential model to device
void sequential_to(Sequential* model, Device device);

// Free memory for a sequential model
void sequential_free(Sequential* model);

#endif // NN_H