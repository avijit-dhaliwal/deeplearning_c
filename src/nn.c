#include "../include/nn.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Linear Layer
typedef struct {
    Module base;
    Tensor* weights;
    Tensor* bias;
    Tensor* input;
    Tensor* output;
} Linear;

static Tensor* linear_forward(Module* module, Tensor* input) {
    Linear* layer = (Linear*)module;
    if (input == NULL) {
        fprintf(stderr, "Error: NULL input tensor in linear forward\n");
        return NULL;
    }
    layer->input = tensor_clone(input);
    if (layer->input == NULL) {
        fprintf(stderr, "Error: Failed to clone input tensor in linear forward\n");
        return NULL;
    }
    layer->output = tensor_matmul(input, layer->weights);
    if (layer->output == NULL) {
        fprintf(stderr, "Error: Matrix multiplication failed in linear forward\n");
        return NULL;
    }
    Tensor* result = tensor_add(layer->output, layer->bias);
    if (result == NULL) {
        fprintf(stderr, "Error: Tensor addition failed in linear forward\n");
        return NULL;
    }
    return result;
}

static void linear_backward(Module* module) {
    Linear* layer = (Linear*)module;
    if (layer->output == NULL || layer->input == NULL) {
        fprintf(stderr, "Error: NULL tensors in linear backward\n");
        return;
    }
    Tensor* grad_output = layer->output;
    layer->base.gradients[0] = tensor_matmul(tensor_transpose(layer->input), grad_output);
    if (layer->base.gradients[0] == NULL) {
        fprintf(stderr, "Error: Failed to compute weight gradients in linear backward\n");
        return;
    }
    layer->base.gradients[1] = tensor_sum_dim(grad_output, 0);
    if (layer->base.gradients[1] == NULL) {
        fprintf(stderr, "Error: Failed to compute bias gradients in linear backward\n");
        return;
    }
    Tensor* grad_input = tensor_matmul(grad_output, tensor_transpose(layer->weights));
    if (grad_input == NULL) {
        fprintf(stderr, "Error: Failed to compute input gradients in linear backward\n");
        return;
    }
    tensor_free(layer->input);
    layer->input = grad_input;
}

static void linear_update(Module* module, float lr) {
    Linear* layer = (Linear*)module;
    Tensor* scaled_grad_weights = tensor_mul_scalar(layer->base.gradients[0], lr);
    Tensor* scaled_grad_bias = tensor_mul_scalar(layer->base.gradients[1], lr);
    if (scaled_grad_weights == NULL || scaled_grad_bias == NULL) {
        fprintf(stderr, "Error: Failed to scale gradients in linear update\n");
        return;
    }
    tensor_sub_inplace(layer->weights, scaled_grad_weights);
    tensor_sub_inplace(layer->bias, scaled_grad_bias);
    tensor_free(scaled_grad_weights);
    tensor_free(scaled_grad_bias);
}

static void linear_to(Module* module, Device device) {
    Linear* layer = (Linear*)module;
    layer->weights = tensor_to(layer->weights, device);
    layer->bias = tensor_to(layer->bias, device);
    if (layer->weights == NULL || layer->bias == NULL) {
        fprintf(stderr, "Error: Failed to transfer linear layer to device\n");
    }
}

static void linear_free(Module* module) {
    Linear* layer = (Linear*)module;
    tensor_free(layer->weights);
    tensor_free(layer->bias);
    tensor_free(layer->input);
    tensor_free(layer->output);
    free(layer->base.parameters);
    free(layer->base.gradients);
    free(layer);
}

Module* nn_linear(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));
    if (layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for linear layer\n");
        return NULL;
    }
    layer->base.forward = linear_forward;
    layer->base.backward = linear_backward;
    layer->base.update = linear_update;
    layer->base.to = linear_to;
    layer->base.free = linear_free;

    layer->base.num_parameters = 2;
    layer->base.parameters = malloc(2 * sizeof(Tensor*));
    layer->base.gradients = malloc(2 * sizeof(Tensor*));
    if (layer->base.parameters == NULL || layer->base.gradients == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for linear layer parameters/gradients\n");
        free(layer);
        return NULL;
    }

    float stddev = sqrt(2.0 / (in_features + out_features));
    layer->weights = tensor_randn((size_t[]){in_features, out_features}, 2, 0, stddev, (Device){CPU, 0});
    layer->bias = tensor_create(NULL, (size_t[]){1, out_features}, 2, (Device){CPU, 0});
    if (layer->weights == NULL || layer->bias == NULL) {
        fprintf(stderr, "Error: Failed to create weight/bias tensors for linear layer\n");
        free(layer->base.parameters);
        free(layer->base.gradients);
        free(layer);
        return NULL;
    }
    tensor_fill_(layer->bias, 0.0f);

    layer->base.parameters[0] = layer->weights;
    layer->base.parameters[1] = layer->bias;

    return (Module*)layer;
}

// Convolutional Layer
typedef struct {
    Module base;
    Tensor* filters;
    Tensor* bias;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    Tensor* input;
    Tensor* output;
} Conv2D;

static Tensor* conv2d_forward(Module* module, Tensor* input) {
    Conv2D* layer = (Conv2D*)module;
    if (input == NULL) {
        fprintf(stderr, "Error: NULL input tensor in conv2d forward\n");
        return NULL;
    }
    layer->input = tensor_clone(input);
    if (layer->input == NULL) {
        fprintf(stderr, "Error: Failed to clone input tensor in conv2d forward\n");
        return NULL;
    }
    
    size_t batch_size = input->shape[0];
    size_t in_height = input->shape[2];
    size_t in_width = input->shape[3];
    size_t out_height = (in_height + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    size_t out_width = (in_width + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    
    Tensor* output = tensor_create(NULL, (size_t[]){batch_size, layer->out_channels, out_height, out_width}, 4, input->device);
    if (output == NULL) {
        fprintf(stderr, "Error: Failed to create output tensor in conv2d forward\n");
        return NULL;
    }
    
    // Implement convolution operation
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oc = 0; oc < layer->out_channels; oc++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    float sum = 0;
                    for (size_t ic = 0; ic < layer->in_channels; ic++) {
                        for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                            for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                                size_t ih = oh * layer->stride + kh - layer->padding;
                                size_t iw = ow * layer->stride + kw - layer->padding;
                                if (ih < in_height && iw < in_width) {
                                    sum += input->data[(b * layer->in_channels + ic) * in_height * in_width + ih * in_width + iw] *
                                           layer->filters->data[(oc * layer->in_channels + ic) * layer->kernel_size * layer->kernel_size + kh * layer->kernel_size + kw];
                                }
                            }
                        }
                    }
                    output->data[(b * layer->out_channels + oc) * out_height * out_width + oh * out_width + ow] = sum + layer->bias->data[oc];
                }
            }
        }
    }
    
    layer->output = output;
    return output;
}

static void conv2d_backward(Module* module) {
    Conv2D* layer = (Conv2D*)module;
    if (layer->output == NULL || layer->input == NULL) {
        fprintf(stderr, "Error: NULL tensors in conv2d backward\n");
        return;
    }
    
    size_t batch_size = layer->input->shape[0];
    size_t in_height = layer->input->shape[2];
    size_t in_width = layer->input->shape[3];
    size_t out_height = layer->output->shape[2];
    size_t out_width = layer->output->shape[3];
    
    layer->base.gradients[0] = tensor_create(NULL, layer->filters->shape, layer->filters->ndim, layer->filters->device);
    layer->base.gradients[1] = tensor_create(NULL, layer->bias->shape, layer->bias->ndim, layer->bias->device);
    if (layer->base.gradients[0] == NULL || layer->base.gradients[1] == NULL) {
        fprintf(stderr, "Error: Failed to create gradient tensors in conv2d backward\n");
        return;
    }
    
    Tensor* grad_input = tensor_create(NULL, layer->input->shape, layer->input->ndim, layer->input->device);
    if (grad_input == NULL) {
        fprintf(stderr, "Error: Failed to create grad_input tensor in conv2d backward\n");
        return;
    }
    
    // Compute gradients
    for (size_t oc = 0; oc < layer->out_channels; oc++) {
        for (size_t ic = 0; ic < layer->in_channels; ic++) {
            for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                    float grad_sum = 0;
                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t oh = 0; oh < out_height; oh++) {
                            for (size_t ow = 0; ow < out_width; ow++) {
                                size_t ih = oh * layer->stride + kh - layer->padding;
                                size_t iw = ow * layer->stride + kw - layer->padding;
                                if (ih < in_height && iw < in_width) {
                                    grad_sum += layer->output->data[(b * layer->out_channels + oc) * out_height * out_width + oh * out_width + ow] *
                                                layer->input->data[(b * layer->in_channels + ic) * in_height * in_width + ih * in_width + iw];
                                }
                            }
                        }
                    }
                    layer->base.gradients[0]->data[(oc * layer->in_channels + ic) * layer->kernel_size * layer->kernel_size + kh * layer->kernel_size + kw] = grad_sum;
                }
            }
        }
    }
    
    for (size_t oc = 0; oc < layer->out_channels; oc++) {
        float grad_sum = 0;
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    grad_sum += layer->output->data[(b * layer->out_channels + oc) * out_height * out_width + oh * out_width + ow];
                }
            }
        }
        layer->base.gradients[1]->data[oc] = grad_sum;
    }
    
    tensor_free(layer->input);
    layer->input = grad_input;
}

static void conv2d_update(Module* module, float lr) {
    Conv2D* layer = (Conv2D*)module;
    Tensor* scaled_grad_filters = tensor_mul_scalar(layer->base.gradients[0], lr);
    Tensor* scaled_grad_bias = tensor_mul_scalar(layer->base.gradients[1], lr);
    if (scaled_grad_filters == NULL || scaled_grad_bias == NULL) {
        fprintf(stderr, "Error: Failed to scale gradients in conv2d update\n");
        return;
    }
    tensor_sub_inplace(layer->filters, scaled_grad_filters);
    tensor_sub_inplace(layer->bias, scaled_grad_bias);
    tensor_free(scaled_grad_filters);
    tensor_free(scaled_grad_bias);
}

static void conv2d_to(Module* module, Device device) {
    Conv2D* layer = (Conv2D*)module;
    layer->filters = tensor_to(layer->filters, device);
    layer->bias = tensor_to(layer->bias, device);
    if (layer->filters == NULL || layer->bias == NULL) {
        fprintf(stderr, "Error: Failed to transfer conv2d layer to device\n");
    }
}

static void conv2d_free(Module* module) {
    Conv2D* layer = (Conv2D*)module;
    tensor_free(layer->filters);
    tensor_free(layer->bias);
    tensor_free(layer->input);
    tensor_free(layer->output);
    free(layer->base.parameters);
    free(layer->base.gradients);
    free(layer);
}

Module* nn_conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding) {
    Conv2D* layer = malloc(sizeof(Conv2D));
    if (layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for conv2d layer\n");
        return NULL;
    }
    layer->base.forward = conv2d_forward;
    layer->base.backward = conv2d_backward;
    layer->base.update = conv2d_update;
    layer->base.to = conv2d_to;
    layer->base.free = conv2d_free;

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    layer->base.num_parameters = 2;
    layer->base.parameters = malloc(2 * sizeof(Tensor*));
    layer->base.gradients = malloc(2 * sizeof(Tensor*));
    if (layer->base.parameters == NULL || layer->base.gradients == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for conv2d layer parameters/gradients\n");
        free(layer);
        return NULL;
    }

    float stddev = sqrt(2.0 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size));
    layer->filters = tensor_randn((size_t[]){out_channels, in_channels, kernel_size, kernel_size}, 4, 0, stddev, (Device){CPU, 0});
    layer->bias = tensor_create(NULL, (size_t[]){out_channels}, 1, (Device){CPU, 0});
    if (layer->filters == NULL || layer->bias == NULL) {
        fprintf(stderr, "Error: Failed to create filter/bias tensors for conv2d layer\n");
        free(layer->base.parameters);
        free(layer->base.gradients);
        free(layer);
        return NULL;
    }
    tensor_fill_(layer->bias, 0.0f);

    layer->base.parameters[0] = layer->filters;
    layer->base.parameters[1] = layer->bias;

    return (Module*)layer;
}

// ReLU Activation
typedef struct {
    Module base;
    Tensor* input;
    Tensor* output;
} ReLU;

static Tensor* relu_forward(Module* module, Tensor* input) {
    ReLU* layer = (ReLU*)module;
    if (input == NULL) {
        fprintf(stderr, "Error: NULL input tensor in ReLU forward\n");
        return NULL;
    }
    layer->input = tensor_clone(input);
    if (layer->input == NULL) {
        fprintf(stderr, "Error: Failed to clone input tensor in ReLU forward\n");
        return NULL;
    }
    layer->output = tensor_create(NULL, input->shape, input->ndim, input->device);
    if (layer->output == NULL) {
        fprintf(stderr, "Error: Failed to create output tensor in ReLU forward\n");
        return NULL;
    }
    for (size_t i = 0; i < input->size; i++) {
        layer->output->data[i] = fmaxf(0, input->data[i]);
    }
    return layer->output;
}

static void relu_backward(Module* module) {
    ReLU* layer = (ReLU*)module;
    if (layer->input == NULL || layer->output == NULL) {
        fprintf(stderr, "Error: NULL tensors in ReLU backward\n");
        return;
    }
    for (size_t i = 0; i < layer->input->size; i++) {
        layer->input->data[i] = layer->input->data[i] > 0 ? layer->output->data[i] : 0;
    }
}

static void relu_update(Module* module, float lr) {
    (void)module;
    (void)lr;
    // ReLU has no parameters to update
}

static void relu_to(Module* module, Device device) {
    ReLU* layer = (ReLU*)module;
    if (layer->input) {
        layer->input = tensor_to(layer->input, device);
    }
    if (layer->output) {
        layer->output = tensor_to(layer->output, device);
    }
}

static void relu_free(Module* module) {
    ReLU* layer = (ReLU*)module;
    tensor_free(layer->input);
    tensor_free(layer->output);
    free(layer);
}

Module* nn_relu() {
    ReLU* layer = malloc(sizeof(ReLU));
    if (layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for ReLU layer\n");
        return NULL;
    }
    layer->base.forward = relu_forward;
    layer->base.backward = relu_backward;
    layer->base.update = relu_update;
    layer->base.to = relu_to;
    layer->base.free = relu_free;
    layer->base.num_parameters = 0;
    layer->base.parameters = NULL;
    layer->base.gradients = NULL;
    layer->input = NULL;
    layer->output = NULL;
    return (Module*)layer;
}

// MaxPool2D Layer
typedef struct {
    Module base;
    size_t kernel_size;
    size_t stride;
    Tensor* input;
    Tensor* output;
    Tensor* max_indices;
} MaxPool2D;

static Tensor* maxpool2d_forward(Module* module, Tensor* input) {
    MaxPool2D* layer = (MaxPool2D*)module;
    if (input == NULL) {
        fprintf(stderr, "Error: NULL input tensor in MaxPool2D forward\n");
        return NULL;
    }
    layer->input = tensor_clone(input);
    if (layer->input == NULL) {
        fprintf(stderr, "Error: Failed to clone input tensor in MaxPool2D forward\n");
        return NULL;
    }
    
    size_t batch_size = input->shape[0];
    size_t channels = input->shape[1];
    size_t in_height = input->shape[2];
    size_t in_width = input->shape[3];
    size_t out_height = (in_height - layer->kernel_size) / layer->stride + 1;
    size_t out_width = (in_width - layer->kernel_size) / layer->stride + 1;
    
    layer->output = tensor_create(NULL, (size_t[]){batch_size, channels, out_height, out_width}, 4, input->device);
    layer->max_indices = tensor_create(NULL, (size_t[]){batch_size, channels, out_height, out_width}, 4, input->device);
    if (layer->output == NULL || layer->max_indices == NULL) {
        fprintf(stderr, "Error: Failed to create output or max_indices tensor in MaxPool2D forward\n");
        return NULL;
    }
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    float max_val = -INFINITY;
                    size_t max_idx = 0;
                    for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                        for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                            size_t ih = oh * layer->stride + kh;
                            size_t iw = ow * layer->stride + kw;
                            size_t idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                            if (input->data[idx] > max_val) {
                                max_val = input->data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    size_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    layer->output->data[out_idx] = max_val;
                    layer->max_indices->data[out_idx] = max_idx;
                }
            }
        }
    }
    
    return layer->output;
}

static void maxpool2d_backward(Module* module) {
    MaxPool2D* layer = (MaxPool2D*)module;
    if (layer->input == NULL || layer->output == NULL || layer->max_indices == NULL) {
        fprintf(stderr, "Error: NULL tensors in MaxPool2D backward\n");
        return;
    }
    
    Tensor* grad_input = tensor_create(NULL, layer->input->shape, layer->input->ndim, layer->input->device);
    if (grad_input == NULL) {
        fprintf(stderr, "Error: Failed to create grad_input tensor in MaxPool2D backward\n");
        return;
    }
    tensor_fill_(grad_input, 0.0f);
    
    size_t output_size = layer->output->size;
    for (size_t i = 0; i < output_size; i++) {
        size_t max_idx = (size_t)layer->max_indices->data[i];
        grad_input->data[max_idx] += layer->output->data[i];
    }
    
    tensor_free(layer->input);
    layer->input = grad_input;
}

static void maxpool2d_update(Module* module, float lr) {
    (void)module;
    (void)lr;
    // MaxPool2D has no parameters to update
}

static void maxpool2d_to(Module* module, Device device) {
    MaxPool2D* layer = (MaxPool2D*)module;
    if (layer->input) {
        layer->input = tensor_to(layer->input, device);
    }
    if (layer->output) {
        layer->output = tensor_to(layer->output, device);
    }
    if (layer->max_indices) {
        layer->max_indices = tensor_to(layer->max_indices, device);
    }
}

static void maxpool2d_free(Module* module) {
    MaxPool2D* layer = (MaxPool2D*)module;
    tensor_free(layer->input);
    tensor_free(layer->output);
    tensor_free(layer->max_indices);
    free(layer);
}

Module* nn_maxpool2d(size_t kernel_size, size_t stride) {
    MaxPool2D* layer = malloc(sizeof(MaxPool2D));
    if (layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for MaxPool2D layer\n");
        return NULL;
    }
    layer->base.forward = maxpool2d_forward;
    layer->base.backward = maxpool2d_backward;
    layer->base.update = maxpool2d_update;
    layer->base.to = maxpool2d_to;
    layer->base.free = maxpool2d_free;
    layer->base.num_parameters = 0;
    layer->base.parameters = NULL;
    layer->base.gradients = NULL;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->input = NULL;
    layer->output = NULL;
    layer->max_indices = NULL;
    return (Module*)layer;
}

// BatchNorm2D Layer
typedef struct {
    Module base;
    size_t num_features;
    float eps;
    float momentum;
    Tensor* gamma;
    Tensor* beta;
    Tensor* running_mean;
    Tensor* running_var;
    Tensor* input;
    Tensor* output;
    Tensor* sample_mean;
    Tensor* sample_var;
    Tensor* normalized;
} BatchNorm2D;

static Tensor* batchnorm2d_forward(Module* module, Tensor* input) {
    BatchNorm2D* layer = (BatchNorm2D*)module;
    if (input == NULL) {
        fprintf(stderr, "Error: NULL input tensor in BatchNorm2D forward\n");
        return NULL;
    }
    layer->input = tensor_clone(input);
    if (layer->input == NULL) {
        fprintf(stderr, "Error: Failed to clone input tensor in BatchNorm2D forward\n");
        return NULL;
    }
    
    size_t batch_size = input->shape[0];
    size_t channels = input->shape[1];
    size_t height = input->shape[2];
    size_t width = input->shape[3];
    
    if (layer->output == NULL) {
        layer->output = tensor_create(NULL, input->shape, input->ndim, input->device);
        if (layer->output == NULL) {
            fprintf(stderr, "Error: Failed to create output tensor in BatchNorm2D forward\n");
            return NULL;
        }
    }
    
    if (layer->base.training) {
        // Calculate mean and variance for each channel
        layer->sample_mean = tensor_create(NULL, (size_t[]){1, channels, 1, 1}, 4, input->device);
        layer->sample_var = tensor_create(NULL, (size_t[]){1, channels, 1, 1}, 4, input->device);
        if (layer->sample_mean == NULL || layer->sample_var == NULL) {
            fprintf(stderr, "Error: Failed to create sample_mean or sample_var tensor in BatchNorm2D forward\n");
            return NULL;
        }
        
        for (size_t c = 0; c < channels; c++) {
            float sum = 0.0f, sq_sum = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        float val = input->data[((b * channels + c) * height + h) * width + w];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }
            float mean = sum / (batch_size * height * width);
            float var = (sq_sum / (batch_size * height * width)) - (mean * mean);
            layer->sample_mean->data[c] = mean;
            layer->sample_var->data[c] = var;
            
            // Update running mean and variance
            layer->running_mean->data[c] = layer->momentum * layer->running_mean->data[c] + (1 - layer->momentum) * mean;
            layer->running_var->data[c] = layer->momentum * layer->running_var->data[c] + (1 - layer->momentum) * var;
        }
    }
    
    // Normalize and scale
    layer->normalized = tensor_create(NULL, input->shape, input->ndim, input->device);
    if (layer->normalized == NULL) {
        fprintf(stderr, "Error: Failed to create normalized tensor in BatchNorm2D forward\n");
        return NULL;
    }
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            float mean = layer->base.training ? layer->sample_mean->data[c] : layer->running_mean->data[c];
            float var = layer->base.training ? layer->sample_var->data[c] : layer->running_var->data[c];
            float std = sqrtf(var + layer->eps);
            float gamma = layer->gamma->data[c];
            float beta = layer->beta->data[c];
            
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = ((b * channels + c) * height + h) * width + w;
                    float normalized = (input->data[idx] - mean) / std;
                    layer->normalized->data[idx] = normalized;
                    layer->output->data[idx] = gamma * normalized + beta;
                }
            }
        }
    }
    
    return layer->output;
}

static void batchnorm2d_backward(Module* module) {
    BatchNorm2D* layer = (BatchNorm2D*)module;
    if (layer->input == NULL || layer->output == NULL || layer->normalized == NULL) {
        fprintf(stderr, "Error: NULL tensors in BatchNorm2D backward\n");
        return;
    }
    
    size_t batch_size = layer->input->shape[0];
    size_t channels = layer->input->shape[1];
    size_t height = layer->input->shape[2];
    size_t width = layer->input->shape[3];
    size_t num_elements = batch_size * height * width;
    
    Tensor* grad_input = tensor_create(NULL, layer->input->shape, layer->input->ndim, layer->input->device);
    layer->base.gradients[0] = tensor_create(NULL, layer->gamma->shape, layer->gamma->ndim, layer->gamma->device);
    layer->base.gradients[1] = tensor_create(NULL, layer->beta->shape, layer->beta->ndim, layer->beta->device);
    
    if (grad_input == NULL || layer->base.gradients[0] == NULL || layer->base.gradients[1] == NULL) {
        fprintf(stderr, "Error: Failed to create gradient tensors in BatchNorm2D backward\n");
        return;
    }
    
    for (size_t c = 0; c < channels; c++) {
        float sum_dy = 0.0f, sum_dy_x = 0.0f;
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = ((b * channels + c) * height + h) * width + w;
                    float dy = layer->output->data[idx];
                    sum_dy += dy;
                    sum_dy_x += dy * layer->normalized->data[idx];
                }
            }
        }
        
        layer->base.gradients[0]->data[c] = sum_dy_x;
        layer->base.gradients[1]->data[c] = sum_dy;
        
        float mean = layer->sample_mean->data[c];
        float var = layer->sample_var->data[c];
        float std = sqrtf(var + layer->eps);
        float gamma = layer->gamma->data[c];
        
        float dx_norm = sum_dy * gamma;
        float dx_var = sum_dy_x * gamma * -0.5f * powf(var + layer->eps, -1.5f);
        float dx_mean = -dx_norm / std - 2.0f * dx_var * mean / num_elements;
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = ((b * channels + c) * height + h) * width + w;
                    grad_input->data[idx] = (dx_norm / std + dx_var * 2.0f * (layer->input->data[idx] - mean) / num_elements + dx_mean / num_elements);
                }
            }
        }
    }
    
    tensor_free(layer->input);
    layer->input = grad_input;
}

static void batchnorm2d_update(Module* module, float lr) {
    BatchNorm2D* layer = (BatchNorm2D*)module;
    Tensor* scaled_grad_gamma = tensor_mul_scalar(layer->base.gradients[0], lr);
    Tensor* scaled_grad_beta = tensor_mul_scalar(layer->base.gradients[1], lr);
    if (scaled_grad_gamma == NULL || scaled_grad_beta == NULL) {
        fprintf(stderr, "Error: Failed to scale gradients in BatchNorm2D update\n");
        return;
    }
    tensor_sub_inplace(layer->gamma, scaled_grad_gamma);
    tensor_sub_inplace(layer->beta, scaled_grad_beta);
    tensor_free(scaled_grad_gamma);
    tensor_free(scaled_grad_beta);
}

static void batchnorm2d_to(Module* module, Device device) {
    BatchNorm2D* layer = (BatchNorm2D*)module;
    layer->gamma = tensor_to(layer->gamma, device);
    layer->beta = tensor_to(layer->beta, device);
    layer->running_mean = tensor_to(layer->running_mean, device);
    layer->running_var = tensor_to(layer->running_var, device);
    if (layer->input) layer->input = tensor_to(layer->input, device);
    if (layer->output) layer->output = tensor_to(layer->output, device);
    if (layer->sample_mean) layer->sample_mean = tensor_to(layer->sample_mean, device);
    if (layer->sample_var) layer->sample_var = tensor_to(layer->sample_var, device);
    if (layer->normalized) layer->normalized = tensor_to(layer->normalized, device);
}

static void batchnorm2d_free(Module* module) {
    BatchNorm2D* layer = (BatchNorm2D*)module;
    tensor_free(layer->gamma);
    tensor_free(layer->beta);
    tensor_free(layer->running_mean);
    tensor_free(layer->running_var);
    tensor_free(layer->input);
    tensor_free(layer->output);
    tensor_free(layer->sample_mean);
    tensor_free(layer->sample_var);
    tensor_free(layer->normalized);
    free(layer->base.parameters);
    free(layer->base.gradients);
    free(layer);
}

Module* nn_batchnorm2d(size_t num_features) {
    BatchNorm2D* layer = malloc(sizeof(BatchNorm2D));
    if (layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for BatchNorm2D layer\n");
        return NULL;
    }
    layer->base.forward = batchnorm2d_forward;
    layer->base.backward = batchnorm2d_backward;
    layer->base.update = batchnorm2d_update;
    layer->base.to = batchnorm2d_to;
    layer->base.free = batchnorm2d_free;
    layer->num_features = num_features;
    layer->eps = 1e-5f;
    layer->momentum = 0.1f;
    
    layer->base.num_parameters = 2;
    layer->base.parameters = malloc(2 * sizeof(Tensor*));
    layer->base.gradients = malloc(2 * sizeof(Tensor*));
    if (layer->base.parameters == NULL || layer->base.gradients == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for BatchNorm2D layer parameters/gradients\n");
        free(layer);
        return NULL;
    }
    
    layer->gamma = tensor_create(NULL, (size_t[]){1, num_features, 1, 1}, 4, (Device){CPU, 0});
    layer->beta = tensor_create(NULL, (size_t[]){1, num_features, 1, 1}, 4, (Device){CPU, 0});
    layer->running_mean = tensor_create(NULL, (size_t[]){1, num_features, 1, 1}, 4, (Device){CPU, 0});
    layer->running_var = tensor_create(NULL, (size_t[]){1, num_features, 1, 1}, 4, (Device){CPU, 0});
    
    if (layer->gamma == NULL || layer->beta == NULL || layer->running_mean == NULL || layer->running_var == NULL) {
        fprintf(stderr, "Error: Failed to create tensors for BatchNorm2D layer\n");
        free(layer->base.parameters);
        free(layer->base.gradients);
        free(layer);
        return NULL;
    }
    
    tensor_fill_(layer->gamma, 1.0f);
    tensor_fill_(layer->beta, 0.0f);
    tensor_fill_(layer->running_mean, 0.0f);
    tensor_fill_(layer->running_var, 1.0f);
    
    layer->base.parameters[0] = layer->gamma;
    layer->base.parameters[1] = layer->beta;
    
    return (Module*)layer;
}

// Sequential model
struct Sequential {
    Module** modules;
    size_t num_modules;
};

Sequential* nn_sequential(Module** modules, size_t num_modules) {
    Sequential* model = malloc(sizeof(Sequential));
    if (model == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Sequential model\n");
        return NULL;
    }
    model->modules = malloc(num_modules * sizeof(Module*));
    if (model->modules == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Sequential model modules\n");
        free(model);
        return NULL;
    }
    memcpy(model->modules, modules, num_modules * sizeof(Module*));
    model->num_modules = num_modules;
    return model;
}

Tensor* sequential_forward(Sequential* model, Tensor* input) {
    Tensor* output = input;
    for (size_t i = 0; i < model->num_modules; i++) {
        output = model->modules[i]->forward(model->modules[i], output);
        if (output == NULL) {
            fprintf(stderr, "Error: Forward pass failed at module %zu\n", i);
            return NULL;
        }
    }
    return output;
}

void sequential_backward(Sequential* model) {
    for (size_t i = model->num_modules; i > 0; i--) {
        model->modules[i-1]->backward(model->modules[i-1]);
    }
}

void sequential_update(Sequential* model, float lr) {
    for (size_t i = 0; i < model->num_modules; i++) {
        model->modules[i]->update(model->modules[i], lr);
    }
}

void sequential_to(Sequential* model, Device device) {
    for (size_t i = 0; i < model->num_modules; i++) {
        model->modules[i]->to(model->modules[i], device);
    }
}

void sequential_free(Sequential* model) {
    for (size_t i = 0; i < model->num_modules; i++) {
        model->modules[i]->free(model->modules[i]);
    }
    free(model->modules);
    free(model);
}