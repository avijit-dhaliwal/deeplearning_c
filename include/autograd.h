#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

typedef struct Node Node;

Node* autograd_variable(Tensor* data);
void autograd_backward(Node* node);
void autograd_zero_grad(Node* node);

#endif // AUTOGRAD_H