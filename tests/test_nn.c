#include "../include/nn.h"
#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_linear_layer() {
    Module* linear = nn_linear(2, 3);
    size_t input_shape[] = {1, 2};
    Tensor* input = tensor_create(NULL, input_shape, 2, (Device){CPU, 0});
    tensor_fill_(input, 1.0f);
    
    Tensor* output = linear->forward(linear, input);
    assert(output != NULL);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 3);
    
    linear->backward(linear);
    linear->update(linear, 0.01f);
    
    tensor_free(input);
    tensor_free(output);
    linear->free(linear);
    printf("test_linear_layer passed\n");
}

void test_relu_activation() {
    Module* relu = nn_relu();
    size_t input_shape[] = {2, 2};
    float input_data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    Tensor* input = tensor_create(input_data, input_shape, 2, (Device){CPU, 0});
    
    Tensor* output = relu->forward(relu, input);
    assert(output != NULL);
    assert(fabs(output->data[0]) < 1e-6);
    assert(fabs(output->data[1]) < 1e-6);
    assert(fabs(output->data[2] - 1.0f) < 1e-6);
    assert(fabs(output->data[3] - 2.0f) < 1e-6);
    
    relu->backward(relu);
    
    tensor_free(input);
    tensor_free(output);
    relu->free(relu);
    printf("test_relu_activation passed\n");
}

int main() {
    test_linear_layer();
    test_relu_activation();
    printf("All neural network tests passed!\n");
    return 0;
}