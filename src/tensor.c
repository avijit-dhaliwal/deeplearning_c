#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

struct Tensor {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    Device device;
#ifdef USE_CUDA
    float* cuda_data;
#endif
};

Tensor* tensor_create(float* data, size_t* shape, size_t ndim, Device device) {
    Tensor* t = malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(size_t));
    t->strides = malloc(ndim * sizeof(size_t));
    memcpy(t->shape, shape, ndim * sizeof(size_t));

    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= shape[i];
    }

    t->device = device;

    if (device.type == CPU) {
        t->data = malloc(t->size * sizeof(float));
        if (data) {
            memcpy(t->data, data, t->size * sizeof(float));
        } else {
            memset(t->data, 0, t->size * sizeof(float));
        }
    }
#ifdef USE_CUDA
    else if (device.type == CUDA) {
        cudaMalloc(&t->cuda_data, t->size * sizeof(float));
        if (data) {
            cudaMemcpy(t->cuda_data, data, t->size * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(t->cuda_data, 0, t->size * sizeof(float));
        }
    }
#endif

    return t;
}

void tensor_free(Tensor* t) {
    if (t->device.type == CPU) {
        free(t->data);
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaFree(t->cuda_data);
    }
#endif
    free(t->shape);
    free(t->strides);
    free(t);
}

Tensor* tensor_to(Tensor* t, Device device) {
    if (t->device.type == device.type) {
        return tensor_clone(t);
    }

    Tensor* new_t = tensor_create(NULL, t->shape, t->ndim, device);

    if (t->device.type == CPU && device.type == CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(new_t->cuda_data, t->data, t->size * sizeof(float), cudaMemcpyHostToDevice);
#endif
    } else if (t->device.type == CUDA && device.type == CPU) {
#ifdef USE_CUDA
        cudaMemcpy(new_t->data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    }

    return new_t;
}

Tensor* tensor_clone(Tensor* t) {
    Tensor* clone = tensor_create(NULL, t->shape, t->ndim, t->device);
    
    if (t->device.type == CPU) {
        memcpy(clone->data, t->data, t->size * sizeof(float));
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaMemcpy(clone->cuda_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
#endif

    return clone;
}

float tensor_item(Tensor* t) {
    if (t->size != 1) {
        fprintf(stderr, "tensor_item only works for tensors with size 1\n");
        return 0;
    }

    float item;
    if (t->device.type == CPU) {
        item = t->data[0];
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaMemcpy(&item, t->cuda_data, sizeof(float), cudaMemcpyDeviceToHost);
    }
#endif

    return item;
}

void tensor_fill_(Tensor* t, float value) {
    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = value;
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        float* h_value = malloc(sizeof(float));
        *h_value = value;
        cudaMemcpy(t->cuda_data, h_value, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(t->cuda_data, value, t->size * sizeof(float));
        free(h_value);
    }
#endif
}

Tensor* tensor_reshape(Tensor* t, size_t* new_shape, size_t new_ndim) {
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }

    if (new_size != t->size) {
        fprintf(stderr, "New shape is incompatible with tensor size\n");
        return NULL;
    }

    Tensor* reshaped = tensor_create(NULL, new_shape, new_ndim, t->device);

    if (t->device.type == CPU) {
        memcpy(reshaped->data, t->data, t->size * sizeof(float));
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaMemcpy(reshaped->cuda_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
#endif

    return reshaped;
}

Tensor* tensor_slice(Tensor* t, size_t start, size_t length) {
    if (t->ndim != 2) {
        fprintf(stderr, "tensor_slice only supports 2D tensors\n");
        return NULL;
    }

    size_t new_shape[2] = {length, t->shape[1]};
    Tensor* slice = tensor_create(NULL, new_shape, 2, t->device);

    if (t->device.type == CPU) {
        memcpy(slice->data, t->data + start * t->shape[1], length * t->shape[1] * sizeof(float));
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaMemcpy(slice->cuda_data, t->cuda_data + start * t->shape[1], length * t->shape[1] * sizeof(float), cudaMemcpyDeviceToDevice);
    }
#endif

    return slice;
}

size_t tensor_argmax(Tensor* t) {
    if (t->ndim != 1) {
        fprintf(stderr, "tensor_argmax only supports 1D tensors\n");
        return 0;
    }

    size_t max_index = 0;
    float max_value;

    if (t->device.type == CPU) {
        max_value = t->data[0];
        for (size_t i = 1; i < t->size; i++) {
            if (t->data[i] > max_value) {
                max_value = t->data[i];
                max_index = i;
            }
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasIsamax(handle, t->size, t->cuda_data, 1, &max_index);
        cublasDestroy(handle);
        max_index--; // CUBLAS returns 1-based index
    }
#endif

    return max_index;
}

float tensor_sum(Tensor* t) {
    float sum = 0;

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            sum += t->data[i];
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSasum(handle, t->size, t->cuda_data, 1, &sum);
        cublasDestroy(handle);
    }
#endif

    return sum;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Tensor sizes do not match for addition\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSaxpy(handle, a->size, &(float){1.0}, a->cuda_data, 1, b->cuda_data, 1);
        cudaMemcpy(result->cuda_data, b->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasDestroy(handle);
    }
#endif

    return result;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Tensor sizes do not match for subtraction\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] - b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSaxpy(handle, a->size, &(float){-1.0}, b->cuda_data, 1, a->cuda_data, 1);
        cudaMemcpy(result->cuda_data, a->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasDestroy(handle);
    }
#endif

    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Tensor sizes do not match for multiplication\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cudaMemcpy(result->cuda_data, a->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, a->size, 1, result->cuda_data, a->size, b->cuda_data, 1, result->cuda_data, a->size);
    }
#endif

    return result;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Tensor sizes do not match for division\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] / b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        // Custom CUDA kernel for element-wise division
        dim3 block(256);
        dim3 grid((a->size + block.x - 1) / block.x);
        elementwise_div<<<grid, block>>>(result->cuda_data, a->cuda_data, b->cuda_data, a->size);
    }
#endif

    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Invalid shapes for matrix multiplication\n");
        return NULL;
    }

    size_t m = a->shape[0];
    size_t n = b->shape[1];
    size_t k = a->shape[1];

    size_t new_shape[2] = {m, n};
    Tensor* result = tensor_create(NULL, new_shape, 2, a->device);

    if (a->device.type == CPU) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0;
                for (size_t p = 0; p < k; p++) {
                    sum += a->data[i * k + p] * b->data[p * n + j];
                }
                result->data[i * n + j] = sum;
            }
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b->cuda_data, n, a->cuda_data, k, &beta, result->cuda_data, n);
        cublasDestroy(handle);
    }
#endif

    return result;
}

Tensor* tensor_transpose(Tensor* t) {
    if (t->ndim != 2) {
        fprintf(stderr, "tensor_transpose only supports 2D tensors\n");
        return NULL;
    }

    size_t new_shape[2] = {t->shape[1], t->shape[0]};
    Tensor* result = tensor_create(NULL, new_shape, 2, t->device);

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->shape[0]; i++) {
            for (size_t j = 0; j < t->shape[1]; j++) {
                result->data[j * t->shape[0] + i] = t->data[i * t->shape[1] + j];
            }
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, t->shape[1], t->shape[0],
                    &alpha, t->cuda_data, t->shape[1],
                    &beta, t->cuda_data, t->shape[1],
                    result->cuda_data, t->shape[0]);
        cublasDestroy(handle);
    }
#endif

    return result;
}

Tensor* tensor_exp(Tensor* t) {
    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = expf(t->data[i]);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise exp
        dim3 block(256);
        dim3 grid((t->size + block.x - 1) / block.x);
        elementwise_exp<<<grid, block>>>(result->cuda_data, t->cuda_data, t->size);
    }
#endif

    return result;
}

Tensor* tensor_log(Tensor* t) {
    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = logf(t->data[i]);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise log
        dim3 block(256);
        dim3 grid((t->size + block.x - 1) / block.x);
        elementwise_log<<<grid, block>>>(result->cuda_data, t->cuda_data, t->size);
    }
#endif

    return result;
}

Tensor* tensor_pow(Tensor* t, float exponent) {
    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = powf(t->data[i], exponent);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise power
        dim3 block(256);
        dim3 grid((t->size + block.x - 1) / block.x);
        elementwise_pow<<<grid, block>>>(result->cuda_data, t->cuda_data, exponent, t->size);
    }
#endif

    return result;
}

Tensor* tensor_mean(Tensor* t, int dim) {
    if (dim < 0 || dim >= t->ndim) {
        fprintf(stderr, "Invalid dimension for mean\n");
        return NULL;
    }

    size_t new_shape[t->ndim - 1];
    size_t j = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != dim) {
            new_shape[j++] = t->shape[i];
        }
    }

    Tensor* result = tensor_create(NULL, new_shape, t->ndim - 1, t->device);

    if (t->device.type == CPU) {
        size_t stride = t->strides[dim];
        size_t dim_size = t->shape[dim];
        for (size_t i = 0; i < result->size; i++) {
            float sum = 0;
            for (size_t k = 0; k < dim_size; k++) {
                sum += t->data[i * stride + k * t->strides[dim]];
            }
            result->data[i] = sum / dim_size;
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for mean reduction
        dim3 block(256);
        dim3 grid((result->size + block.x - 1) / block.x);
        mean_reduction<<<grid, block>>>(result->cuda_data, t->cuda_data, t->shape[dim], t->strides[dim], result->size);
    }
#endif

    return result;
}

Tensor* tensor_sum_dim(Tensor* t, int dim) {
    if (dim < 0 || dim >= t->ndim) {
        fprintf(stderr, "Invalid dimension for sum\n");
        return NULL;
    }

    size_t new_shape[t->ndim - 1];
    size_t j = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != dim) {
            new_shape[j++] = t->shape[i];
        }
    }

    Tensor* result = tensor_create(NULL, new_shape, t->ndim - 1, t->device);

    if (t->device.type == CPU) {
        size_t stride = t->strides[dim];
        size_t dim_size = t->shape[dim];
        for (size_t i = 0; i < result->size; i++) {
            float sum = 0;
            for (size_t k = 0; k < dim_size; k++) {
                sum += t->data[i * stride + k * t->strides[dim]];
            }
            result->data[i] = sum;
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for sum reduction
        dim3 block(256);
        dim3 grid((result->size + block.x - 1) / block.x);
        sum_reduction<<<grid, block>>>(result->cuda_data, t->cuda_data, t->shape[dim], t->strides[dim], result->size);
    }
#endif

    return result;
}

void tensor_print(Tensor* t) {
    printf("Tensor shape: (");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");

    if (t->device.type == CUDA) {
        printf("Device: CUDA\n");
        return;
    }

    if (t->ndim == 1) {
        for (size_t i = 0; i < t->shape[0]; i++) {
            printf("%f ", t->data[i]);
        }
        printf("\n");
    } else if (t->ndim == 2) {
        for (size_t i = 0; i < t->shape[0]; i++) {
            for (size_t j = 0; j < t->shape[1]; j++) {
                printf("%f ", t->data[i * t->shape[1] + j]);
            }
            printf("\n");
        }
    } else {
        printf("Tensor data: ");
        for (size_t i = 0; i < t->size; i++) {
            printf("%f ", t->data[i]);
        }
        printf("\n");
    }
}