#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_tensor_create() {
    size_t shape[] = {2, 3};
    Tensor* t = tensor_create(NULL, shape, 2, (Device){CPU, 0});
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->size == 6);
    tensor_free(t);
    printf("test_tensor_create passed\n");
}

void test_tensor_fill() {
    size_t shape[] = {2, 3};
    Tensor* t = tensor_create(NULL, shape, 2, (Device){CPU, 0});
    tensor_fill_(t, 3.14f);
    for (size_t i = 0; i < t->size; i++) {
        assert(fabs(t->data[i] - 3.14f) < 1e-6);
    }
    tensor_free(t);
    printf("test_tensor_fill passed\n");
}

void test_tensor_add() {
    size_t shape[] = {2, 3};
    float data1[] = {1, 2, 3, 4, 5, 6};
    float data2[] = {6, 5, 4, 3, 2, 1};
    Tensor* t1 = tensor_create(data1, shape, 2, (Device){CPU, 0});
    Tensor* t2 = tensor_create(data2, shape, 2, (Device){CPU, 0});
    Tensor* result = tensor_add(t1, t2);
    for (size_t i = 0; i < result->size; i++) {
        assert(fabs(result->data[i] - 7.0f) < 1e-6);
    }
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(result);
    printf("test_tensor_add passed\n");
}

void test_tensor_matmul() {
    size_t shape1[] = {2, 3};
    size_t shape2[] = {3, 2};
    float data1[] = {1, 2, 3, 4, 5, 6};
    float data2[] = {7, 8, 9, 10, 11, 12};
    Tensor* t1 = tensor_create(data1, shape1, 2, (Device){CPU, 0});
    Tensor* t2 = tensor_create(data2, shape2, 2, (Device){CPU, 0});
    Tensor* result = tensor_matmul(t1, t2);
    assert(result->shape[0] == 2);
    assert(result->shape[1] == 2);
    float expected[] = {58, 64, 139, 154};
    for (size_t i = 0; i < result->size; i++) {
        assert(fabs(result->data[i] - expected[i]) < 1e-6);
    }
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(result);
    printf("test_tensor_matmul passed\n");
}

int main() {
    test_tensor_create();
    test_tensor_fill();
    test_tensor_add();
    test_tensor_matmul();
    printf("All tensor tests passed!\n");
    return 0;
}