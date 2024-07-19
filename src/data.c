#include "../include/data.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Dataset {
    Tensor* features;
    Tensor* labels;
    size_t num_samples;
};

Dataset* dataset_create(Tensor* features, Tensor* labels) {
    if (!features || !labels) {
        fprintf(stderr, "Error: NULL input tensors in dataset creation\n");
        return NULL;
    }
    if (features->shape[0] != labels->shape[0]) {
        fprintf(stderr, "Error: Mismatched number of samples in features and labels\n");
        return NULL;
    }

    Dataset* dataset = malloc(sizeof(Dataset));
    if (!dataset) {
        fprintf(stderr, "Error: Failed to allocate memory for dataset\n");
        return NULL;
    }
    dataset->features = features;
    dataset->labels = labels;
    dataset->num_samples = features->shape[0];
    return dataset;
}

void dataset_free(Dataset* dataset) {
    if (dataset) {
        tensor_free(dataset->features);
        tensor_free(dataset->labels);
        free(dataset);
    }
}

struct DataLoader {
    Dataset* dataset;
    size_t batch_size;
    bool shuffle;
    size_t* indices;
    size_t current_index;
};

DataLoader* dataloader_create(Dataset* dataset, size_t batch_size, bool shuffle) {
    if (!dataset) {
        fprintf(stderr, "Error: NULL dataset in dataloader creation\n");
        return NULL;
    }
    if (batch_size == 0 || batch_size > dataset->num_samples) {
        fprintf(stderr, "Error: Invalid batch size in dataloader creation\n");
        return NULL;
    }

    DataLoader* dataloader = malloc(sizeof(DataLoader));
    if (!dataloader) {
        fprintf(stderr, "Error: Failed to allocate memory for dataloader\n");
        return NULL;
    }
    dataloader->dataset = dataset;
    dataloader->batch_size = batch_size;
    dataloader->shuffle = shuffle;
    dataloader->indices = malloc(dataset->num_samples * sizeof(size_t));
    if (!dataloader->indices) {
        fprintf(stderr, "Error: Failed to allocate memory for dataloader indices\n");
        free(dataloader);
        return NULL;
    }
    for (size_t i = 0; i < dataset->num_samples; i++) {
        dataloader->indices[i] = i;
    }
    dataloader->current_index = 0;
    
    if (shuffle) {
        srand(time(NULL));
        for (size_t i = dataset->num_samples - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t temp = dataloader->indices[i];
            dataloader->indices[i] = dataloader->indices[j];
            dataloader->indices[j] = temp;
        }
    }
    
    return dataloader;
}

void dataloader_reset(DataLoader* dataloader) {
    if (!dataloader) {
        fprintf(stderr, "Error: NULL dataloader in reset\n");
        return;
    }
    dataloader->current_index = 0;
    if (dataloader->shuffle) {
        for (size_t i = dataloader->dataset->num_samples - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t temp = dataloader->indices[i];
            dataloader->indices[i] = dataloader->indices[j];
            dataloader->indices[j] = temp;
        }
    }
}

int dataloader_next(DataLoader* dataloader, Tensor** batch_features, Tensor** batch_labels) {
    if (!dataloader || !batch_features || !batch_labels) {
        fprintf(stderr, "Error: NULL input in dataloader next\n");
        return 0;
    }
    
    if (dataloader->current_index >= dataloader->dataset->num_samples) {
        return 0;  // End of epoch
    }
    
    size_t remaining_samples = dataloader->dataset->num_samples - dataloader->current_index;
    size_t current_batch_size = (remaining_samples < dataloader->batch_size) ? remaining_samples : dataloader->batch_size;
    
    size_t* feature_shape = malloc((dataloader->dataset->features->ndim) * sizeof(size_t));
    size_t* label_shape = malloc((dataloader->dataset->labels->ndim) * sizeof(size_t));
    
    if (!feature_shape || !label_shape) {
        fprintf(stderr, "Error: Failed to allocate memory for shapes in dataloader next\n");
        free(feature_shape);
        free(label_shape);
        return 0;
    }
    
    memcpy(feature_shape, dataloader->dataset->features->shape, dataloader->dataset->features->ndim * sizeof(size_t));
    memcpy(label_shape, dataloader->dataset->labels->shape, dataloader->dataset->labels->ndim * sizeof(size_t));
    
    feature_shape[0] = current_batch_size;
    label_shape[0] = current_batch_size;
    
    *batch_features = tensor_create(NULL, feature_shape, dataloader->dataset->features->ndim, dataloader->dataset->features->device);
    *batch_labels = tensor_create(NULL, label_shape, dataloader->dataset->labels->ndim, dataloader->dataset->labels->device);
    
    if (!*batch_features || !*batch_labels) {
        fprintf(stderr, "Error: Failed to create batch tensors in dataloader next\n");
        tensor_free(*batch_features);
        tensor_free(*batch_labels);
        free(feature_shape);
        free(label_shape);
        return 0;
    }
    
    for (size_t i = 0; i < current_batch_size; i++) {
        size_t idx = dataloader->indices[dataloader->current_index + i];
        tensor_copy_slice(dataloader->dataset->features, *batch_features, idx, i);
        tensor_copy_slice(dataloader->dataset->labels, *batch_labels, idx, i);
    }
    
    dataloader->current_index += current_batch_size;
    
    free(feature_shape);
    free(label_shape);
    
    return 1;  // Successful batch creation
}

void dataloader_free(DataLoader* dataloader) {
    if (dataloader) {
        free(dataloader->indices);
        free(dataloader);
    }
}