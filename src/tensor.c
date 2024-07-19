// tensor.c

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
    if (shape == NULL || ndim == 0) {
        fprintf(stderr, "Error: Invalid shape or dimensions\n");
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Tensor\n");
        return NULL;
    }

    t->ndim = ndim;
    t->shape = (size_t*)malloc(ndim * sizeof(size_t));
    t->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (t->shape == NULL || t->strides == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for shape or strides\n");
        tensor_free(t);
        return NULL;
    }

    memcpy(t->shape, shape, ndim * sizeof(size_t));

    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= shape[i];
    }

    t->device = device;

    if (device.type == CPU) {
        t->data = (float*)malloc(t->size * sizeof(float));
        if (t->data == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for tensor data\n");
            tensor_free(t);
            return NULL;
        }
        if (data) {
            memcpy(t->data, data, t->size * sizeof(float));
        } else {
            memset(t->data, 0, t->size * sizeof(float));
        }
    }
#ifdef USE_CUDA
    else if (device.type == CUDA) {
        cudaError_t cuda_status = cudaMalloc(&t->cuda_data, t->size * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memory allocation failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(t);
            return NULL;
        }
        if (data) {
            cuda_status = cudaMemcpy(t->cuda_data, data, t->size * sizeof(float), cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "Error: CUDA memcpy failed: %s\n", cudaGetErrorString(cuda_status));
                tensor_free(t);
                return NULL;
            }
        } else {
            cuda_status = cudaMemset(t->cuda_data, 0, t->size * sizeof(float));
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "Error: CUDA memset failed: %s\n", cudaGetErrorString(cuda_status));
                tensor_free(t);
                return NULL;
            }
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(t);
        return NULL;
    }

    return t;
}

void tensor_free(Tensor* t) {
    if (t == NULL) return;

    free(t->shape);
    free(t->strides);

    if (t->device.type == CPU) {
        free(t->data);
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaFree(t->cuda_data);
    }
#endif

    free(t);
}

Tensor* tensor_clone(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot clone NULL tensor\n");
        return NULL;
    }

    Tensor* clone = tensor_create(NULL, t->shape, t->ndim, t->device);
    if (clone == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        memcpy(clone->data, t->data, t->size * sizeof(float));
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaError_t cuda_status = cudaMemcpy(clone->cuda_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in clone: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(clone);
            return NULL;
        }
    }
#endif

    return clone;
}

Tensor* tensor_to(Tensor* t, Device device) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot transfer NULL tensor\n");
        return NULL;
    }

    if (t->device.type == device.type) {
        return tensor_clone(t);
    }

    Tensor* new_t = tensor_create(NULL, t->shape, t->ndim, device);
    if (new_t == NULL) {
        return NULL;
    }

    if (t->device.type == CPU && device.type == CUDA) {
#ifdef USE_CUDA
        cudaError_t cuda_status = cudaMemcpy(new_t->cuda_data, t->data, t->size * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in to: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(new_t);
            return NULL;
        }
#else
        fprintf(stderr, "Error: CUDA support not enabled\n");
        tensor_free(new_t);
        return NULL;
#endif
    } else if (t->device.type == CUDA && device.type == CPU) {
#ifdef USE_CUDA
        cudaError_t cuda_status = cudaMemcpy(new_t->data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in to: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(new_t);
            return NULL;
        }
#else
        fprintf(stderr, "Error: CUDA support not enabled\n");
        tensor_free(new_t);
        return NULL;
#endif
    }

    return new_t;
}

float tensor_item(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot get item from NULL tensor\n");
        return 0.0f;
    }

    if (t->size != 1) {
        fprintf(stderr, "Error: tensor_item only works for tensors with size 1\n");
        return 0.0f;
    }

    float item;
    if (t->device.type == CPU) {
        item = t->data[0];
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaError_t cuda_status = cudaMemcpy(&item, t->cuda_data, sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in item: %s\n", cudaGetErrorString(cuda_status));
            return 0.0f;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        return 0.0f;
    }

    return item;
}

void tensor_fill_(Tensor* t, float value) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot fill NULL tensor\n");
        return;
    }

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = value;
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaError_t cuda_status = cudaMemset(t->cuda_data, *(int*)&value, t->size * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memset failed in fill: %s\n", cudaGetErrorString(cuda_status));
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
    }
}

Tensor* tensor_reshape(Tensor* t, size_t* new_shape, size_t new_ndim) {
    if (t == NULL || new_shape == NULL) {
        fprintf(stderr, "Error: Invalid input for reshape\n");
        return NULL;
    }

    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }

    if (new_size != t->size) {
        fprintf(stderr, "Error: New shape is incompatible with tensor size\n");
        return NULL;
    }

    Tensor* reshaped = tensor_create(NULL, new_shape, new_ndim, t->device);
    if (reshaped == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        memcpy(reshaped->data, t->data, t->size * sizeof(float));
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cudaError_t cuda_status = cudaMemcpy(reshaped->cuda_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in reshape: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(reshaped);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(reshaped);
        return NULL;
    }

    return reshaped;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot add NULL tensors\n");
        return NULL;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for addition\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);
    if (result == NULL) {
        return NULL;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        cublasSaxpy(handle, a->size, &alpha, a->cuda_data, 1, b->cuda_data, 1);
        cudaMemcpy(result->cuda_data, b->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot subtract NULL tensors\n");
        return NULL;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for subtraction\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);
    if (result == NULL) {
        return NULL;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] - b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = -1.0f;
        cublasSaxpy(handle, a->size, &alpha, b->cuda_data, 1, a->cuda_data, 1);
        cudaMemcpy(result->cuda_data, a->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot multiply NULL tensors\n");
        return NULL;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for multiplication\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);
    if (result == NULL) {
        return NULL;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cudaError_t cuda_status = cudaMemcpy(result->cuda_data, a->cuda_data, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in mul: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, a->size, 1, &(float){1.0f}, b->cuda_data, 1, result->cuda_data, 1, &(float){0.0f}, result->cuda_data, 1);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot divide NULL tensors\n");
        return NULL;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for division\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, a->shape, a->ndim, a->device);
    if (result == NULL) {
        return NULL;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            if (b->data[i] == 0) {
                fprintf(stderr, "Error: Division by zero\n");
                tensor_free(result);
                return NULL;
            }
            result->data[i] = a->data[i] / b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        // Custom CUDA kernel for element-wise division
        dim3 block_size(256);
        dim3 grid_size((a->size + block_size.x - 1) / block_size.x);
        cuda_div_kernel<<<grid_size, block_size>>>(result->cuda_data, a->cuda_data, b->cuda_data, a->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot perform matrix multiplication on NULL tensors\n");
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: Invalid shapes for matrix multiplication\n");
        return NULL;
    }

    size_t m = a->shape[0];
    size_t n = b->shape[1];
    size_t k = a->shape[1];

    size_t new_shape[2] = {m, n};
    Tensor* result = tensor_create(NULL, new_shape, 2, a->device);
    if (result == NULL) {
        return NULL;
    }

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
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_transpose(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot transpose NULL tensor\n");
        return NULL;
    }

    if (t->ndim != 2) {
        fprintf(stderr, "Error: transpose only supports 2D tensors\n");
        return NULL;
    }

    size_t new_shape[2] = {t->shape[1], t->shape[0]};
    Tensor* result = tensor_create(NULL, new_shape, 2, t->device);
    if (result == NULL) {
        return NULL;
    }

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
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_exp(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot compute exp of NULL tensor\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);
    if (result == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = expf(t->data[i]);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise exp
        dim3 block_size(256);
        dim3 grid_size((t->size + block_size.x - 1) / block_size.x);
        cuda_exp_kernel<<<grid_size, block_size>>>(result->cuda_data, t->cuda_data, t->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_log(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot compute log of NULL tensor\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);
    if (result == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            if (t->data[i] <= 0) {
                fprintf(stderr, "Error: Invalid input for log\n");
                tensor_free(result);
                return NULL;
            }
            result->data[i] = logf(t->data[i]);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise log
        dim3 block_size(256);
        dim3 grid_size((t->size + block_size.x - 1) / block_size.x);
        cuda_log_kernel<<<grid_size, block_size>>>(result->cuda_data, t->cuda_data, t->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_pow(Tensor* t, float exponent) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot compute power of NULL tensor\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);
    if (result == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = powf(t->data[i], exponent);
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        // Custom CUDA kernel for element-wise power
        dim3 block_size(256);
        dim3 grid_size((t->size + block_size.x - 1) / block_size.x);
        cuda_pow_kernel<<<grid_size, block_size>>>(result->cuda_data, t->cuda_data, exponent, t->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_mean(Tensor* t, int dim) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot compute mean of NULL tensor\n");
        return NULL;
    }

    if (dim < 0 || dim >= t->ndim) {
        fprintf(stderr, "Error: Invalid dimension for mean\n");
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
    if (result == NULL) {
        return NULL;
    }

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
        dim3 block_size(256);
        dim3 grid_size((result->size + block_size.x - 1) / block_size.x);
        cuda_mean_reduction<<<grid_size, block_size>>>(result->cuda_data, t->cuda_data, t->shape[dim], t->strides[dim], result->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

Tensor* tensor_sum(Tensor* t, int dim) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot compute sum of NULL tensor\n");
        return NULL;
    }

    if (dim < 0 || dim >= t->ndim) {
        fprintf(stderr, "Error: Invalid dimension for sum\n");
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
    if (result == NULL) {
        return NULL;
    }

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
        dim3 block_size(256);
        dim3 grid_size((result->size + block_size.x - 1) / block_size.x);
        cuda_sum_reduction<<<grid_size, block_size>>>(result->cuda_data, t->cuda_data, t->shape[dim], t->strides[dim], result->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            tensor_free(result);
            return NULL;
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

void tensor_print(Tensor* t) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot print NULL tensor\n");
        return;
    }

    printf("Tensor shape: (");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");

    if (t->device.type == CUDA) {
        printf("Device: CUDA\n");
        // Copy data to CPU for printing
        float* cpu_data = (float*)malloc(t->size * sizeof(float));
        if (cpu_data == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for printing CUDA tensor\n");
            return;
        }
        cudaError_t cuda_status = cudaMemcpy(cpu_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA memcpy failed in print: %s\n", cudaGetErrorString(cuda_status));
            free(cpu_data);
            return;
        }
        
        // Print the data
        if (t->ndim == 1) {
            for (size_t i = 0; i < t->shape[0]; i++) {
                printf("%f ", cpu_data[i]);
            }
            printf("\n");
        } else if (t->ndim == 2) {
            for (size_t i = 0; i < t->shape[0]; i++) {
                for (size_t j = 0; j < t->shape[1]; j++) {
                    printf("%f ", cpu_data[i * t->shape[1] + j]);
                }
                printf("\n");
            }
        } else {
            printf("Tensor data: ");
            for (size_t i = 0; i < t->size; i++) {
                printf("%f ", cpu_data[i]);
            }
            printf("\n");
        }
        
        free(cpu_data);
    } else {
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
}

Tensor* tensor_randn(size_t* shape, size_t ndim, float mean, float stddev, Device device) {
    Tensor* t = tensor_create(NULL, shape, ndim, device);
    if (t == NULL) {
        return NULL;
    }

    if (device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            t->data[i] = mean + stddev * z;
        }
    }
#ifdef USE_CUDA
    else if (device.type == CUDA) {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        curandGenerateNormal(gen, t->cuda_data, t->size, mean, stddev);
        curandDestroyGenerator(gen);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(t);
        return NULL;
    }

    return t;
}

Tensor* tensor_mul_scalar(Tensor* t, float scalar) {
    if (t == NULL) {
        fprintf(stderr, "Error: Cannot multiply NULL tensor with scalar\n");
        return NULL;
    }

    Tensor* result = tensor_create(NULL, t->shape, t->ndim, t->device);
    if (result == NULL) {
        return NULL;
    }

    if (t->device.type == CPU) {
        for (size_t i = 0; i < t->size; i++) {
            result->data[i] = t->data[i] * scalar;
        }
    }
#ifdef USE_CUDA
    else if (t->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSscal(handle, t->size, &scalar, t->cuda_data, 1);
        cudaMemcpy(result->cuda_data, t->cuda_data, t->size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
        tensor_free(result);
        return NULL;
    }

    return result;
}

void tensor_add_inplace(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot perform in-place addition with NULL tensor(s)\n");
        return;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for in-place addition\n");
        return;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] += b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        cublasSaxpy(handle, a->size, &alpha, b->cuda_data, 1, a->cuda_data, 1);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
    }
}

void tensor_sub_inplace(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot perform in-place subtraction with NULL tensor(s)\n");
        return;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for in-place subtraction\n");
        return;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] -= b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = -1.0f;
        cublasSaxpy(handle, a->size, &alpha, b->cuda_data, 1, a->cuda_data, 1);
        cublasDestroy(handle);
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
    }
}

void tensor_mul_inplace(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot perform in-place multiplication with NULL tensor(s)\n");
        return;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for in-place multiplication\n");
        return;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] *= b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        // Custom CUDA kernel for element-wise multiplication
        dim3 block_size(256);
        dim3 grid_size((a->size + block_size.x - 1) / block_size.x);
        cuda_mul_inplace<<<grid_size, block_size>>>(a->cuda_data, b->cuda_data, a->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
    }
}

void tensor_div_inplace(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Cannot perform in-place division with NULL tensor(s)\n");
        return;
    }

    if (a->size != b->size) {
        fprintf(stderr, "Error: Tensor sizes do not match for in-place division\n");
        return;
    }

    if (a->device.type == CPU) {
        for (size_t i = 0; i < a->size; i++) {
            if (b->data[i] == 0) {
                fprintf(stderr, "Error: Division by zero\n");
                return;
            }
            a->data[i] /= b->data[i];
        }
    }
#ifdef USE_CUDA
    else if (a->device.type == CUDA) {
        // Custom CUDA kernel for element-wise division
        dim3 block_size(256);
        dim3 grid_size((a->size + block_size.x - 1) / block_size.x);
        cuda_div_inplace<<<grid_size, block_size>>>(a->cuda_data, b->cuda_data, a->size);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        }
    }
#endif
    else {
        fprintf(stderr, "Error: Unsupported device type\n");
    }
}

#ifdef USE_CUDA
// CUDA kernel implementations
__global__ void cuda_div_kernel(float* result, float* a, float* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] != 0) {
            result[idx] = a[idx] / b[idx];
        } else {
            result[idx] = 0;  // or some other error handling
        }
    }
}

__global__ void cuda_exp_kernel(float* result, float* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = expf(input[idx]);
    }
}

__global__ void cuda_log_kernel(float* result, float* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0) {
            result[idx] = logf(input[idx]);
        } else {
            result[idx] = -INFINITY;  // or some other error handling
        }
    }
}

__global__ void cuda_pow_kernel(float* result, float* input, float exponent, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = powf(input[idx], exponent);
    }
}

__global__ void cuda_mean_reduction(float* result, float* input, size_t dim_size, size_t stride, size_t result_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < result_size) {
        float sum = 0;
        for (size_t i = 0; i < dim_size; i++) {
            sum += input[idx * stride + i * stride];
        }
        result[idx] = sum / dim_size;
    }
}

__global__ void cuda_sum_reduction(float* result, float* input, size_t dim_size, size_t stride, size_t result_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < result_size) {
        float sum = 0;
        for (size_t i = 0; i < dim_size; i++) {
            sum += input[idx * stride + i * stride];
        }
        result[idx] = sum;
    }
}

__global__ void cuda_mul_inplace(float* a, float* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

__global__ void cuda_div_inplace(float* a, float* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] != 0) {
            a[idx] /= b[idx];
        } else {
            a[idx] = 0;  // or some other error handling
        }
    }
}
#endif