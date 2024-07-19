#ifndef SERIALIZE_H
#define SERIALIZE_H

#include "nn.h"

void save_model(Sequential* model, const char* filename);
void save_checkpoint(Sequential* model, Optimizer* optimizer, int epoch, float loss, const char* filename);
void load_checkpoint(Sequential** model, Optimizer** optimizer, int* epoch, float* loss, const char* filename);
Sequential* load_model(const char* filename);

#endif // SERIALIZE_H