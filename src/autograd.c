#include "../include/autograd.h"
#include <stdlib.h>

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MATMUL,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_SOFTMAX
} OperationType;

struct Node {
    Tensor* data;
    Tensor* grad;
    OperationType op;
    Node** children;
    size_t num_children;
    void (*backward)(struct Node*);
};

static void add_backward(Node* node) {
    for (size_t i = 0; i < node->num_children; i++) {
        Node* child = node->children[i];
        for (size_t j = 0; j < child->data->size; j++) {
            child->grad->data[j] += node->grad->data[j];
        }
    }
}

static void mul_backward(Node* node) {
    Node* a = node->children[0];
    Node* b = node->children[1];
    for (size_t i = 0; i < a->data->size; i++) {
        a->grad->data[i] += node->grad->data[i] * b->data->data[i];
        b->grad->data[i] += node->grad->data[i] * a->data->data[i];
    }
}

static void matmul_backward(Node* node) {
    Node* a = node->children[0];
    Node* b = node->children[1];
    
    // dC/dA = dC/dY * B^T
    Tensor* b_t = tensor_transpose(b->data);
    Tensor* da = tensor_matmul(node->grad, b_t);
    tensor_add_inplace(a->grad, da);
    tensor_free(da);
    tensor_free(b_t);
    
    // dC/dB = A^T * dC/dY
    Tensor* a_t = tensor_transpose(a->data);
    Tensor* db = tensor_matmul(a_t, node->grad);
    tensor_add_inplace(b->grad, db);
    tensor_free(db);
    tensor_free(a_t);
}

static void relu_backward(Node* node) {
    Node* input = node->children[0];
    for (size_t i = 0; i < input->data->size; i++) {
        input->grad->data[i] += (input->data->data[i] > 0) ? node->grad->data[i] : 0;
    }
}

static void sigmoid_backward(Node* node) {
    Node* input = node->children[0];
    for (size_t i = 0; i < input->data->size; i++) {
        float sigmoid_x = 1 / (1 + exp(-input->data->data[i]));
        input->grad->data[i] += node->grad->data[i] * sigmoid_x * (1 - sigmoid_x);
    }
}

static void tanh_backward(Node* node) {
    Node* input = node->children[0];
    for (size_t i = 0; i < input->data->size; i++) {
        float tanh_x = tanh(input->data->data[i]);
        input->grad->data[i] += node->grad->data[i] * (1 - tanh_x * tanh_x);
    }
}

static void softmax_backward(Node* node) {
    Node* input = node->children[0];
    size_t batch_size = input->data->shape[0];
    size_t num_classes = input->data->shape[1];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < num_classes; j++) {
            float sj = node->data->data[i * num_classes + j];
            for (size_t k = 0; k < num_classes; k++) {
                float sk = node->data->data[i * num_classes + k];
                float grad = (j == k) ? sj * (1 - sj) : -sj * sk;
                input->grad->data[i * num_classes + j] += node->grad->data[i * num_classes + k] * grad;
            }
        }
    }
}

Node* autograd_variable(Tensor* data) {
    Node* node = malloc(sizeof(Node));
    node->data = data;
    node->grad = tensor_create(NULL, data->shape, data->ndim);
    node->op = OP_ADD;  // Dummy operation for leaf nodes
    node->children = NULL;
    node->num_children = 0;
    node->backward = NULL;
    return node;
}

Node* autograd_add(Node* a, Node* b) {
    Node* result = malloc(sizeof(Node));
    result->data = tensor_add(a->data, b->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_ADD;
    result->children = malloc(2 * sizeof(Node*));
    result->children[0] = a;
    result->children[1] = b;
    result->num_children = 2;
    result->backward = add_backward;
    return result;
}

Node* autograd_mul(Node* a, Node* b) {
    Node* result = malloc(sizeof(Node));
    result->data = tensor_multiply(a->data, b->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_MUL;
    result->children = malloc(2 * sizeof(Node*));
    result->children[0] = a;
    result->children[1] = b;
    result->num_children = 2;
    result->backward = mul_backward;
    return result;
}

Node* autograd_matmul(Node* a, Node* b) {
    Node* result = malloc(sizeof(Node));
    result->data = tensor_matmul(a->data, b->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_MATMUL;
    result->children = malloc(2 * sizeof(Node*));
    result->children[0] = a;
    result->children[1] = b;
    result->num_children = 2;
    result->backward = matmul_backward;
    return result;
}

Node* autograd_relu(Node* input) {
    Node* result = malloc(sizeof(Node));
    result->data = ops_relu(input->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_RELU;
    result->children = malloc(sizeof(Node*));
    result->children[0] = input;
    result->num_children = 1;
    result->backward = relu_backward;
    return result;
}

Node* autograd_sigmoid(Node* input) {
    Node* result = malloc(sizeof(Node));
    result->data = ops_sigmoid(input->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_SIGMOID;
    result->children = malloc(sizeof(Node*));
    result->children[0] = input;
    result->num_children = 1;
    result->backward = sigmoid_backward;
    return result;
}

Node* autograd_tanh(Node* input) {
    Node* result = malloc(sizeof(Node));
    result->data = ops_tanh(input->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_TANH;
    result->children = malloc(sizeof(Node*));
    result->children[0] = input;
    result->num_children = 1;
    result->backward = tanh_backward;
    return result;
}

Node* autograd_softmax(Node* input) {
    Node* result = malloc(sizeof(Node));
    result->data = ops_softmax(input->data);
    result->grad = tensor_create(NULL, result->data->shape, result->data->ndim);
    result->op = OP_SOFTMAX;
    result->children = malloc(sizeof(Node*));
    result->children[0] = input;
    result->num_children = 1;
    result->backward = softmax_backward;
    return result;
}

void autograd_backward(Node* node) {
    // Set gradient of output to 1
    tensor_fill(node->grad, 1.0);
    
    // Perform topological sort
    Node** topo_order = malloc(1000 * sizeof(Node*));  // Assume max 1000 nodes
    size_t topo_size = 0;
    
    void topo_sort(Node* n) {
        if (n->backward) {
            for (size_t i = 0; i < n->num_children; i++) {
                topo_sort(n->children[i]);
            }
            topo_order[topo_size++] = n;
        }
    }
    
    topo_sort(node);
    
    // Backward pass
    for (size_t i = topo_size; i > 0; i--) {
        Node* n = topo_order[i - 1];
        n->backward(n);
    }
    
    free(topo_order);
}

void autograd_zero_grad(Node* node) {
    tensor_fill(node->grad, 0.0);
    for (size_t i = 0; i < node->num_children; i++) {
        autograd_zero_grad(node->children[i]);
    }
}