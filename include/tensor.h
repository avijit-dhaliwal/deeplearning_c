#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    CPU,
    CUDA
} DeviceType;

typedef struct {
    DeviceType type;
    int device_index;
} Device;

typedef struct Tensor Tensor;

Tensor* tensor_create(float* data, size_t* shape, size_t ndim, Device device);
void tensor_free(Tensor* t);
Tensor* tensor_to(Tensor* t, Device device);
Tensor* tensor_clone(Tensor* t);
float tensor_item(Tensor* t);
void tensor_fill_(Tensor* t, float value);
Tensor* tensor_reshape(Tensor* t, size_t* new_shape, size_t new_ndim);
Tensor* tensor_slice(Tensor* t, size_t start, size_t length);
size_t tensor_argmax(Tensor* t);
float tensor_sum(Tensor* t);
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_transpose(Tensor* t);
Tensor* tensor_exp(Tensor* t);
Tensor* tensor_log(Tensor* t);
Tensor* tensor_pow(Tensor* t, float exponent);
Tensor* tensor_mean(Tensor* t, int dim);
Tensor* tensor_sum_dim(Tensor* t, int dim);
void tensor_print(Tensor* t);

// New functions
Tensor* tensor_randn(size_t* shape, size_t ndim, float mean, float stddev, Device device);
Tensor* tensor_mul_scalar(Tensor* t, float scalar);
void tensor_sub_inplace(Tensor* a, Tensor* b);

#endif // TENSOR_H