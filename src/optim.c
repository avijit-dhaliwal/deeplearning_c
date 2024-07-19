#include "../include/optim.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct Optimizer {
    OptimizerType type;
    Sequential* model;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    Tensor** m;
    Tensor** v;
};

Optimizer* optim_sgd(Sequential* model, float learning_rate) {
    Optimizer* optim = malloc(sizeof(Optimizer));
    if (!optim) {
        fprintf(stderr, "Error: Failed to allocate memory for SGD optimizer\n");
        return NULL;
    }
    optim->type = OPTIM_SGD;
    optim->model = model;
    optim->learning_rate = learning_rate;
    optim->m = NULL;
    optim->v = NULL;
    return optim;
}

Optimizer* optim_adam(Sequential* model, float learning_rate, float beta1, float beta2, float epsilon) {
    Optimizer* optim = malloc(sizeof(Optimizer));
    if (!optim) {
        fprintf(stderr, "Error: Failed to allocate memory for Adam optimizer\n");
        return NULL;
    }
    optim->type = OPTIM_ADAM;
    optim->model = model;
    optim->learning_rate = learning_rate;
    optim->beta1 = beta1;
    optim->beta2 = beta2;
    optim->epsilon = epsilon;
    optim->t = 0;
    
    size_t num_params = 0;
    for (size_t i = 0; i < model->num_modules; i++) {
        num_params += model->modules[i]->num_parameters;
    }
    
    optim->m = malloc(num_params * sizeof(Tensor*));
    optim->v = malloc(num_params * sizeof(Tensor*));
    if (!optim->m || !optim->v) {
        fprintf(stderr, "Error: Failed to allocate memory for Adam optimizer parameters\n");
        free(optim->m);
        free(optim->v);
        free(optim);
        return NULL;
    }
    
    size_t param_index = 0;
    for (size_t i = 0; i < model->num_modules; i++) {
        for (size_t j = 0; j < model->modules[i]->num_parameters; j++) {
            Tensor* param = model->modules[i]->parameters[j];
            optim->m[param_index] = tensor_create(NULL, param->shape, param->ndim, param->device);
            optim->v[param_index] = tensor_create(NULL, param->shape, param->ndim, param->device);
            if (!optim->m[param_index] || !optim->v[param_index]) {
                fprintf(stderr, "Error: Failed to create tensor for Adam optimizer\n");
                for (size_t k = 0; k <= param_index; k++) {
                    tensor_free(optim->m[k]);
                    tensor_free(optim->v[k]);
                }
                free(optim->m);
                free(optim->v);
                free(optim);
                return NULL;
            }
            tensor_fill_(optim->m[param_index], 0.0f);
            tensor_fill_(optim->v[param_index], 0.0f);
            param_index++;
        }
    }
    
    return optim;
}

void optimizer_step(Optimizer* optim) {
    if (optim->type == OPTIM_SGD) {
        for (size_t i = 0; i < optim->model->num_modules; i++) {
            Module* module = optim->model->modules[i];
            for (size_t j = 0; j < module->num_parameters; j++) {
                Tensor* param = module->parameters[j];
                Tensor* grad = module->gradients[j];
                Tensor* update = tensor_mul_scalar(grad, optim->learning_rate);
                tensor_sub_inplace(param, update);
                tensor_free(update);
            }
        }
    } else if (optim->type == OPTIM_ADAM) {
        optim->t++;
        float lr_t = optim->learning_rate * sqrtf(1.0f - powf(optim->beta2, optim->t)) / (1.0f - powf(optim->beta1, optim->t));
        
        size_t param_index = 0;
        for (size_t i = 0; i < optim->model->num_modules; i++) {
            Module* module = optim->model->modules[i];
            for (size_t j = 0; j < module->num_parameters; j++) {
                Tensor* param = module->parameters[j];
                Tensor* grad = module->gradients[j];
                Tensor* m = optim->m[param_index];
                Tensor* v = optim->v[param_index];
                
                // m = beta1 * m + (1 - beta1) * grad
                Tensor* m_scaled = tensor_mul_scalar(m, optim->beta1);
                Tensor* grad_scaled = tensor_mul_scalar(grad, 1.0f - optim->beta1);
                tensor_add_inplace(m_scaled, grad_scaled);
                tensor_copy_(m, m_scaled);
                tensor_free(m_scaled);
                tensor_free(grad_scaled);
                
                // v = beta2 * v + (1 - beta2) * grad^2
                Tensor* v_scaled = tensor_mul_scalar(v, optim->beta2);
                Tensor* grad_squared = tensor_mul(grad, grad);
                Tensor* grad_squared_scaled = tensor_mul_scalar(grad_squared, 1.0f - optim->beta2);
                tensor_add_inplace(v_scaled, grad_squared_scaled);
                tensor_copy_(v, v_scaled);
                tensor_free(v_scaled);
                tensor_free(grad_squared);
                tensor_free(grad_squared_scaled);
                
                // param = param - lr_t * m / (sqrt(v) + epsilon)
                Tensor* v_sqrt = tensor_pow(v, 0.5f);
                tensor_add_scalar_inplace(v_sqrt, optim->epsilon);
                Tensor* update = tensor_div(m, v_sqrt);
                tensor_mul_scalar_inplace(update, lr_t);
                tensor_sub_inplace(param, update);
                
                tensor_free(v_sqrt);
                tensor_free(update);
                
                param_index++;
            }
        }
    }
}

void optimizer_zero_grad(Optimizer* optim) {
    for (size_t i = 0; i < optim->model->num_modules; i++) {
        Module* module = optim->model->modules[i];
        for (size_t j = 0; j < module->num_parameters; j++) {
            Tensor* grad = module->gradients[j];
            tensor_fill_(grad, 0.0f);
        }
    }
}

void optimizer_free(Optimizer* optim) {
    if (optim->type == OPTIM_ADAM) {
        size_t num_params = 0;
        for (size_t i = 0; i < optim->model->num_modules; i++) {
            num_params += optim->model->modules[i]->num_parameters;
        }
        for (size_t i = 0; i < num_params; i++) {
            tensor_free(optim->m[i]);
            tensor_free(optim->v[i]);
        }
        free(optim->m);
        free(optim->v);
    }
    free(optim);
}