#include "../include/optim.h"
#include "../include/nn.h"
#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_sgd_optimizer() {
    Module* linear = nn_linear(2, 1);
    Module* layers[] = {linear};
    Sequential* model = nn_sequential(layers, 1);
    
    Optimizer* sgd = optim_sgd(model, 0.1f);
    
    size_t input_shape[] = {1, 2};
    Tensor* input = tensor_create(NULL, input_shape, 2, (Device){CPU, 0});
    tensor_fill_(input, 1.0f);
    
    Tensor* output = sequential_forward(model, input);
    sequential_backward(model);
    optimizer_step(sgd);
    
    tensor_free(input);
    tensor_free(output);
    sequential_free(model);
    optimizer_free(sgd);
    printf("test_sgd_optimizer passed\n");
}

void test_adam_optimizer() {
    Module* linear = nn_linear(2, 1);
    Module* layers[] = {linear};
    Sequential* model = nn_sequential(layers, 1);
    
    Optimizer* adam = optim_adam(model, 0.001f, 0.9f, 0.999f, 1e-8f);
    
    size_t input_shape[] = {1, 2};
    Tensor* input = tensor_create(NULL, input_shape, 2, (Device){CPU, 0});
    tensor_fill_(input, 1.0f);
    
    Tensor* output = sequential_forward(model, input);
    sequential_backward(model);
    optimizer_step(adam);
    
    tensor_free(input);
    tensor_free(output);
    sequential_free(model);
    optimizer_free(adam);
    printf("test_adam_optimizer passed\n");
}

int main() {
    test_sgd_optimizer();
    test_adam_optimizer();
    printf("All optimizer tests passed!\n");
    return 0;
}