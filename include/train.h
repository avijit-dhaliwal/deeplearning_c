#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"
#include "optim.h"
#include "loss.h"
#include "data.h"

typedef struct {
    float train_loss;
    float train_accuracy;
    float val_loss;
    float val_accuracy;
} TrainResult;

TrainResult train_epoch(Sequential* model, Optimizer* optimizer, Loss* criterion, DataLoader* train_loader, DataLoader* val_loader);
float evaluate(Sequential* model, Loss* criterion, DataLoader* data_loader);

#endif // TRAIN_H