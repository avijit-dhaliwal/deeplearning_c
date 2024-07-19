#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

typedef struct Optimizer Optimizer;

Optimizer* optim_sgd(Tensor** params, size_t num_params, float learning_rate);
Optimizer* optim_adam(Tensor** params, size_t num_params, float learning_rate, float beta1, float beta2, float eps);
void optimizer_step(Optimizer* optim);
void optimizer_zero_grad(Optimizer* optim);
void optimizer_free(Optimizer* optim);

#endif // OPTIM_H