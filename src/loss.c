#include "../include/loss.h"
#include <stdlib.h>
#include <math.h>

typedef enum {
    MSE,
    CROSS_ENTROPY
} LossType;

struct Loss {
    LossType type;
    Tensor* output;
};

Loss* loss_mse() {
    Loss* loss = malloc(sizeof(Loss));
    loss->type = MSE;
    loss->output = NULL;
    return loss;
}

Loss* loss_cross_entropy() {
    Loss* loss = malloc(sizeof(Loss));
    loss->type = CROSS_ENTROPY;
    loss->output = NULL;
    return loss;
}

Tensor* loss_forward(Loss* loss, Tensor* predictions, Tensor* targets) {
    if (loss->type == MSE) {
        Tensor* diff = tensor_sub(predictions, targets);
        Tensor* squared_diff = tensor_mul(diff, diff);
        loss->output = tensor_mean(squared_diff, -1);
        tensor_free(diff);
        tensor_free(squared_diff);
    } else if (loss->type == CROSS_ENTROPY) {
        Tensor* log_predictions = tensor_log(predictions);
        Tensor* neg_log_pred = tensor_mul_scalar(log_predictions, -1.0f);
        Tensor* product = tensor_mul(targets, neg_log_pred);
        loss->output = tensor_sum_dim(product, -1);
        tensor_free(log_predictions);
        tensor_free(neg_log_pred);
        tensor_free(product);
    }
    return loss->output;
}

Tensor* loss_backward(Loss* loss, Tensor* predictions, Tensor* targets) {
    Tensor* grad;
    if (loss->type == MSE) {
        grad = tensor_sub(predictions, targets);
        tensor_mul_scalar_inplace(grad, 2.0f / predictions->shape[0]);
    } else if (loss->type == CROSS_ENTROPY) {
        Tensor* eps = tensor_create(NULL, predictions->shape, predictions->ndim, predictions->device);
        tensor_fill_(eps, 1e-7f);
        Tensor* pred_plus_eps = tensor_add(predictions, eps);
        grad = tensor_div(targets, pred_plus_eps);
        tensor_mul_scalar_inplace(grad, -1.0f / predictions->shape[0]);
        tensor_free(eps);
        tensor_free(pred_plus_eps);
    }
    return grad;
}

void loss_free(Loss* loss) {
    if (loss->output != NULL) {
        tensor_free(loss->output);
    }
    free(loss);
}