#include "../include/serialize.h"
#include <stdio.h>

void save_model(Sequential* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&model->num_modules, sizeof(size_t), 1, file);
    
    for (size_t i = 0; i < model->num_modules; i++) {
        Module* module = model->modules[i];
        fwrite(&module->type, sizeof(int), 1, file);
        
        switch (module->type) {
            case MODULE_LINEAR:
            case MODULE_CONV2D:
            case MODULE_BATCHNORM2D:
                for (size_t j = 0; j < module->num_parameters; j++) {
                    Tensor* param = module->parameters[j];
                    fwrite(&param->ndim, sizeof(size_t), 1, file);
                    fwrite(param->shape, sizeof(size_t), param->ndim, file);
                    fwrite(param->data, sizeof(float), param->size, file);
                }
                break;
            case MODULE_RELU:
            case MODULE_MAXPOOL2D:
            case MODULE_DROPOUT:
                // These layers don't have parameters to save
                break;
            default:
                fprintf(stderr, "Error: Unknown module type when saving\n");
                fclose(file);
                return;
        }
    }
    
    fclose(file);
}

Sequential* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file for reading: %s\n", filename);
        return NULL;
    }
    
    size_t num_modules;
    fread(&num_modules, sizeof(size_t), 1, file);
    
    Module** modules = malloc(num_modules * sizeof(Module*));
    if (!modules) {
        fprintf(stderr, "Error: Failed to allocate memory for modules\n");
        fclose(file);
        return NULL;
    }
    
    for (size_t i = 0; i < num_modules; i++) {
        int module_type;
        fread(&module_type, sizeof(int), 1, file);
        
        Module* module;
        switch (module_type) {
            case MODULE_LINEAR: {
                size_t in_features, out_features;
                fread(&in_features, sizeof(size_t), 1, file);
                fread(&out_features, sizeof(size_t), 1, file);
                module = nn_linear(in_features, out_features);
                
                // Load weights
                fread(((Linear*)module)->weights->data, sizeof(float), in_features * out_features, file);
                // Load bias
                fread(((Linear*)module)->bias->data, sizeof(float), out_features, file);
                break;
            }
            case MODULE_CONV2D: {
                size_t in_channels, out_channels, kernel_size, stride, padding;
                fread(&in_channels, sizeof(size_t), 1, file);
                fread(&out_channels, sizeof(size_t), 1, file);
                fread(&kernel_size, sizeof(size_t), 1, file);
                fread(&stride, sizeof(size_t), 1, file);
                fread(&padding, sizeof(size_t), 1, file);
                module = nn_conv2d(in_channels, out_channels, kernel_size, stride, padding);
                
                // Load filters
                fread(((Conv2D*)module)->filters->data, sizeof(float), out_channels * in_channels * kernel_size * kernel_size, file);
                // Load bias
                fread(((Conv2D*)module)->bias->data, sizeof(float), out_channels, file);
                break;
            }
            case MODULE_RELU:
                module = nn_relu();
                break;
            case MODULE_MAXPOOL2D: {
                size_t kernel_size, stride;
                fread(&kernel_size, sizeof(size_t), 1, file);
                fread(&stride, sizeof(size_t), 1, file);
                module = nn_maxpool2d(kernel_size, stride);
                break;
            }
            case MODULE_BATCHNORM2D: {
                size_t num_features;
                fread(&num_features, sizeof(size_t), 1, file);
                module = nn_batchnorm2d(num_features);
                
                // Load gamma
                fread(((BatchNorm2D*)module)->gamma->data, sizeof(float), num_features, file);
                // Load beta
                fread(((BatchNorm2D*)module)->beta->data, sizeof(float), num_features, file);
                // Load running_mean
                fread(((BatchNorm2D*)module)->running_mean->data, sizeof(float), num_features, file);
                // Load running_var
                fread(((BatchNorm2D*)module)->running_var->data, sizeof(float), num_features, file);
                break;
            }
            case MODULE_DROPOUT: {
                float p;
                fread(&p, sizeof(float), 1, file);
                module = nn_dropout(p);
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown module type when loading\n");
                fclose(file);
                for (size_t j = 0; j < i; j++) {
                    modules[j]->free(modules[j]);
                }
                free(modules);
                return NULL;
        }
        
        modules[i] = module;
    }
    
    fclose(file);
    
    return nn_sequential(modules, num_modules);
}