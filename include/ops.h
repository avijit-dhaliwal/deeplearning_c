#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor* ops_relu(Tensor* input);
Tensor* ops_sigmoid(Tensor* input);
Tensor* ops_tanh(Tensor* input);
Tensor* ops_softmax(Tensor* input);

#endif // OPS_H