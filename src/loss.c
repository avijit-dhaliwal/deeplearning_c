#include "../include/loss.h"
#include <stdlib.h>
#include <math.h>

struct Loss {
    LossType type;
    Tensor* output;
};

Loss* loss_mse() {
    Loss* loss = malloc(sizeof(Loss));
    if (!loss) {
        fprintf(stderr, "Error: Failed to allocate memory for MSE loss\n");
        return NULL;
    }
    loss->type = LOSS_MSE;
    loss->output = NULL;
    return loss;
}

Loss* loss_cross_entropy() {
    Loss* loss = malloc(sizeof(Loss));
    if (!loss) {
        fprintf(stderr, "Error: Failed to allocate memory for Cross Entropy loss\n");
        return NULL;
    }
    loss->type = LOSS_CROSS_ENTROPY;
    loss->output = NULL;
    return loss;
}

Tensor* loss_forward(Loss* loss, Tensor* predictions, Tensor* targets) {
    if (!predictions || !targets) {
        fprintf(stderr, "Error: NULL input tensors in loss forward\n");
        return NULL;
    }

    if (loss->type == LOSS_MSE) {
        Tensor* diff = tensor_sub(predictions, targets);
        if (!diff) {
            fprintf(stderr, "Error: Failed to compute difference in MSE loss\n");
            return NULL;
        }
        Tensor* squared_diff = tensor_mul(diff, diff);
        if (!squared_diff) {
            fprintf(stderr, "Error: Failed to compute squared difference in MSE loss\n");
            tensor_free(diff);
            return NULL;
        }
        loss->output = tensor_mean(squared_diff, -1);
        if (!loss->output) {
            fprintf(stderr, "Error: Failed to compute mean in MSE loss\n");
            tensor_free(diff);
            tensor_free(squared_diff);
            return NULL;
        }
        tensor_free(diff);
        tensor_free(squared_diff);
    } else if (loss->type == LOSS_CROSS_ENTROPY) {
        Tensor* log_predictions = tensor_log(predictions);
        if (!log_predictions) {
            fprintf(stderr, "Error: Failed to compute log in Cross Entropy loss\n");
            return NULL;
        }
        Tensor* neg_log_pred = tensor_mul_scalar(log_predictions, -1.0f);
        if (!neg_log_pred) {
            fprintf(stderr, "Error: Failed to compute negative log in Cross Entropy loss\n");
            tensor_free(log_predictions);
            return NULL;
        }
        Tensor* product = tensor_mul(targets, neg_log_pred);
        if (!product) {
            fprintf(stderr, "Error: Failed to compute product in Cross Entropy loss\n");
            tensor_free(log_predictions);
            tensor_free(neg_log_pred);
            return NULL;
        }
        loss->output = tensor_sum_dim(product, -1);
        if (!loss->output) {
            fprintf(stderr, "Error: Failed to compute sum in Cross Entropy loss\n");
            tensor_free(log_predictions);
            tensor_free(neg_log_pred);
            tensor_free(product);
            return NULL;
        }
        tensor_free(log_predictions);
        tensor_free(neg_log_pred);
        tensor_free(product);
    }
    return loss->output;
}

Tensor* loss_backward(Loss* loss, Tensor* predictions, Tensor* targets) {
    if (!predictions || !targets) {
        fprintf(stderr, "Error: NULL input tensors in loss backward\n");
        return NULL;
    }

    Tensor* grad;
    if (loss->type == LOSS_MSE) {
        grad = tensor_sub(predictions, targets);
        if (!grad) {
            fprintf(stderr, "Error: Failed to compute gradient in MSE loss backward\n");
            return NULL;
        }
        tensor_mul_scalar_inplace(grad, 2.0f / predictions->shape[0]);
    } else if (loss->type == LOSS_CROSS_ENTROPY) {
        Tensor* eps = tensor_create(NULL, predictions->shape, predictions->ndim, predictions->device);
        if (!eps) {
            fprintf(stderr, "Error: Failed to create epsilon tensor in Cross Entropy loss backward\n");
            return NULL;
        }
        tensor_fill_(eps, 1e-7f);
        Tensor* pred_plus_eps = tensor_add(predictions, eps);
        if (!pred_plus_eps) {
            fprintf(stderr, "Error: Failed to add epsilon in Cross Entropy loss backward\n");
            tensor_free(eps);
            return NULL;
        }
        grad = tensor_div(targets, pred_plus_eps);
        if (!grad) {
            fprintf(stderr, "Error: Failed to compute gradient in Cross Entropy loss backward\n");
            tensor_free(eps);
            tensor_free(pred_plus_eps);
            return NULL;
        }
        tensor_mul_scalar_inplace(grad, -1.0f / predictions->shape[0]);
        tensor_free(eps);
        tensor_free(pred_plus_eps);
    } else {
        fprintf(stderr, "Error: Unknown loss type\n");
        return NULL;
    }
    return grad;
}

void loss_free(Loss* loss) {
    if (loss->output) {
        tensor_free(loss->output);
    }
    free(loss);
}