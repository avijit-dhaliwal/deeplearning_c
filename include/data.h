#ifndef DATA_H
#define DATA_H

#include "tensor.h"
#include <stdbool.h>

typedef struct Dataset Dataset;
typedef struct DataLoader DataLoader;

Dataset* dataset_create(Tensor* features, Tensor* labels);
void dataset_free(Dataset* dataset);

DataLoader* dataloader_create(Dataset* dataset, size_t batch_size, bool shuffle);
void dataloader_reset(DataLoader* dataloader);
int dataloader_next(DataLoader* dataloader, Tensor** batch_features, Tensor** batch_labels);
void dataloader_free(DataLoader* dataloader);

#endif // DATA_H