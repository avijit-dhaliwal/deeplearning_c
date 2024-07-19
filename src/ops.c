#include "../include/ops.h"
#include <math.h>

Tensor* ops_relu(Tensor* input) {
    Tensor* result = tensor_create(NULL, input->shape, input->ndim);
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
    return result;
}

Tensor* ops_sigmoid(Tensor* input) {
    Tensor* result = tensor_create(NULL, input->shape, input->ndim);
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = 1 / (1 + exp(-input->data[i]));
    }
    return result;
}

Tensor* ops_tanh(Tensor* input) {
    Tensor* result = tensor_create(NULL, input->shape, input->ndim);
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = tanh(input->data[i]);
    }
    return result;
}

Tensor* ops_softmax(Tensor* input) {
    if (input->ndim != 2) {
        fprintf(stderr, "Softmax only supports 2D tensors\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, input->shape, input->ndim);
    size_t batch_size = input->shape[0];
    size_t class_count = input->shape[1];

    for (size_t i = 0; i < batch_size; i++) {
        float max_val = input->data[i * class_count];
        for (size_t j = 1; j < class_count; j++) {
            if (input->data[i * class_count + j] > max_val) {
                max_val = input->data[i * class_count + j];
            }
        }

        float sum = 0;
        for (size_t j = 0; j < class_count; j++) {
            float exp_val = exp(input->data[i * class_count + j] - max_val);
            result->data[i * class_count + j] = exp_val;
            sum += exp_val;
        }

        for (size_t j = 0; j < class_count; j++) {
            result->data[i * class_count + j] /= sum;
        }
    }

    return result;
}