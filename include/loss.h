#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef struct Loss Loss;

Loss* loss_mse();
Loss* loss_cross_entropy();
Tensor* loss_forward(Loss* loss, Tensor* predictions, Tensor* targets);
Tensor* loss_backward(Loss* loss, Tensor* predictions, Tensor* targets);
void loss_free(Loss* loss);

#endif // LOSS_H