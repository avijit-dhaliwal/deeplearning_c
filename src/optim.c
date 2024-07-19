#include "../include/optim.h"
#include <stdlib.h>
#include <math.h>

typedef enum {
    OPTIM_SGD,
    OPTIM_ADAM
} OptimizerType;

struct Optimizer {
    OptimizerType type;
    Tensor** params;
    size_t num_params;
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    Tensor** m;
    Tensor** v;
    size_t t;
};

Optimizer* optim_sgd(Tensor** params, size_t num_params, float learning_rate) {
    Optimizer* optim = malloc(sizeof(Optimizer));
    optim->type = OPTIM_SGD;
    optim->params = params;
    optim->num_params = num_params;
    optim->learning_rate = learning_rate;
    return optim;
}

Optimizer* optim_adam(Tensor** params, size_t num_params, float learning_rate, float beta1, float beta2, float eps) {
    Optimizer* optim = malloc(sizeof(Optimizer));
    optim->type = OPTIM_ADAM;
    optim->params = params;
    optim->num_params = num_params;
    optim->learning_rate = learning_rate;
    optim->beta1 = beta1;
    optim->beta2 = beta2;
    optim->eps = eps;
    optim->t = 0;
    
    optim->m = malloc(num_params * sizeof(Tensor*));
    optim->v = malloc(num_params * sizeof(Tensor*));
    for (size_t i = 0; i < num_params; i++) {
        optim->m[i] = tensor_create(NULL, params[i]->shape, params[i]->ndim);
        optim->v[i] = tensor_create(NULL, params[i]->shape, params[i]->ndim);
    }
    
    return optim;
}

static void sgd_step(Optimizer* optim) {
    for (size_t i = 0; i < optim->num_params; i++) {
        Tensor* param = optim->params[i];
        for (size_t j = 0; j < param->size; j++) {
            param->data[j] -= optim->learning_rate * param->grad[j];
        }
    }
}

static void adam_step(Optimizer* optim) {
    optim->t++;
    float lr_t = optim->learning_rate * sqrt(1 - pow(optim->beta2, optim->t)) / (1 - pow(optim->beta1, optim->t));
    
    for (size_t i = 0; i < optim->num_params; i++) {
        Tensor* param = optim->params[i];
        Tensor* m = optim->m[i];
        Tensor* v = optim->v[i];
        
        for (size_t j = 0; j < param->size; j++) {
            m->data[j] = optim->beta1 * m->data[j] + (1 - optim->beta1) * param->grad[j];
            v->data[j] = optim->beta2 * v->data[j] + (1 - optim->beta2) * param->grad[j] * param->grad[j];
            param->data[j] -= lr_t * m->data[j] / (sqrt(v->data[j]) + optim->eps);
        }
    }
}

void optimizer_step(Optimizer* optim) {
    switch (optim->type) {
        case OPTIM_SGD:
            sgd_step(optim);
            break;
        case OPTIM_ADAM:
            adam_step(optim);
            break;
        default:
            fprintf(stderr, "Unknown optimizer type\n");
    }
}

void optimizer_zero_grad(Optimizer* optim) {
    for (size_t i = 0; i < optim->num_params; i++) {
        Tensor* param = optim->params[i];
        for (size_t j = 0; j < param->size; j++) {
            param->grad[j] = 0;
        }
    }
}

void optimizer_free(Optimizer* optim) {
    if (optim->type == OPTIM_ADAM) {
        for (size_t i = 0; i < optim->num_params; i++) {
            tensor_free(optim->m[i]);
            tensor_free(optim->v[i]);
        }
        free(optim->m);
        free(optim->v);
    }
    free(optim);
}