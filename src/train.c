#include "../include/train.h"
#include <stdio.h>

TrainResult train_epoch(Sequential* model, Optimizer* optimizer, Loss* criterion, DataLoader* train_loader, DataLoader* val_loader) {
    TrainResult result = {0};
    size_t correct = 0;
    size_t total = 0;

    // Training
    dataloader_reset(train_loader);
    Tensor *batch_features, *batch_labels;
    while (dataloader_next(train_loader, &batch_features, &batch_labels)) {
        optimizer_zero_grad(optimizer);
        
        Tensor* outputs = sequential_forward(model, batch_features);
        if (!outputs) {
            fprintf(stderr, "Error: Forward pass failed in training\n");
            tensor_free(batch_features);
            tensor_free(batch_labels);
            return result;
        }
        
        Tensor* loss = loss_forward(criterion, outputs, batch_labels);
        if (!loss) {
            fprintf(stderr, "Error: Loss computation failed in training\n");
            tensor_free(outputs);
            tensor_free(batch_features);
            tensor_free(batch_labels);
            return result;
        }
        
        result.train_loss += tensor_item(loss);
        
        Tensor* grad = loss_backward(criterion, outputs, batch_labels);
        if (!grad) {
            fprintf(stderr, "Error: Backward pass failed in training\n");
            tensor_free(loss);
            tensor_free(outputs);
            tensor_free(batch_features);
            tensor_free(batch_labels);
            return result;
        }
        
        sequential_backward(model);
        optimizer_step(optimizer);
        
        // Calculate accuracy
        size_t batch_size = batch_features->shape[0];
        for (size_t i = 0; i < batch_size; i++) {
            if (tensor_argmax(tensor_slice(outputs, i, 1)) == tensor_argmax(tensor_slice(batch_labels, i, 1))) {
                correct++;
            }
            total++;
        }
        
        tensor_free(outputs);
        tensor_free(loss);
        tensor_free(grad);
        tensor_free(batch_features);
        tensor_free(batch_labels);
    }
    
    result.train_loss /= train_loader->dataset->num_samples;
    result.train_accuracy = (float)correct / total;
    
    // Validation
    if (val_loader) {
        result.val_loss = evaluate(model, criterion, val_loader);
        result.val_accuracy = (float)correct / total;
    }
    
    return result;
}

float evaluate(Sequential* model, Loss* criterion, DataLoader* data_loader) {
    float total_loss = 0;
    size_t correct = 0;
    size_t total = 0;
    
    dataloader_reset(data_loader);
    Tensor *batch_features, *batch_labels;
    while (dataloader_next(data_loader, &batch_features, &batch_labels)) {
        Tensor* outputs = sequential_forward(model, batch_features);
        if (!outputs) {
            fprintf(stderr, "Error: Forward pass failed in evaluation\n");
            tensor_free(batch_features);
            tensor_free(batch_labels);
            return 0;
        }
        
        Tensor* loss = loss_forward(criterion, outputs, batch_labels);
        if (!loss) {
            fprintf(stderr, "Error: Loss computation failed in evaluation\n");
            tensor_free(outputs);
            tensor_free(batch_features);
            tensor_free(batch_labels);
            return 0;
        }
        
        total_loss += tensor_item(loss);
        
        // Calculate accuracy
        size_t batch_size = batch_features->shape[0];
        for (size_t i = 0; i < batch_size; i++) {
            if (tensor_argmax(tensor_slice(outputs, i, 1)) == tensor_argmax(tensor_slice(batch_labels, i, 1))) {
                correct++;
            }
            total++;
        }
        
        tensor_free(outputs);
        tensor_free(loss);
        tensor_free(batch_features);
        tensor_free(batch_labels);
    }
    
    return total_loss / data_loader->dataset->num_samples;
}