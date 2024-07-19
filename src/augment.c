#include "../include/augment.h"
#include <stdlib.h>
#include <math.h>

Tensor* augment_random_crop(Tensor* image, int crop_size) {
    int height = image->shape[1];
    int width = image->shape[2];
    
    int top = rand() % (height - crop_size + 1);
    int left = rand() % (width - crop_size + 1);
    
    size_t new_shape[] = {image->shape[0], crop_size, crop_size};
    Tensor* cropped = tensor_create(NULL, new_shape, 3, image->device);
    
    for (int c = 0; c < image->shape[0]; c++) {
        for (int i = 0; i < crop_size; i++) {
            for (int j = 0; j < crop_size; j++) {
                int src_idx = c * height * width + (top + i) * width + (left + j);
                int dst_idx = c * crop_size * crop_size + i * crop_size + j;
                cropped->data[dst_idx] = image->data[src_idx];
            }
        }
    }
    
    return cropped;
}

Tensor* augment_random_flip(Tensor* image) {
    if (rand() % 2 == 0) {
        return tensor_clone(image);
    }
    
    Tensor* flipped = tensor_create(NULL, image->shape, image->ndim, image->device);
    int height = image->shape[1];
    int width = image->shape[2];
    
    for (int c = 0; c < image->shape[0]; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int src_idx = c * height * width + i * width + j;
                int dst_idx = c * height * width + i * width + (width - 1 - j);
                flipped->data[dst_idx] = image->data[src_idx];
            }
        }
    }
    
    return flipped;
}

Tensor* augment_random_rotation(Tensor* image, float max_angle) {
    float angle = ((float)rand() / RAND_MAX) * 2 * max_angle - max_angle;
    float radian = angle * M_PI / 180.0;
    float cos_theta = cos(radian);
    float sin_theta = sin(radian);
    
    int height = image->shape[1];
    int width = image->shape[2];
    int channels = image->shape[0];
    
    Tensor* rotated = tensor_create(NULL, image->shape, image->ndim, image->device);
    
    int center_x = width / 2;
    int center_y = height / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int new_x = (int)((x - center_x) * cos_theta - (y - center_y) * sin_theta + center_x);
            int new_y = (int)((x - center_x) * sin_theta + (y - center_y) * cos_theta + center_y);
            
            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                for (int c = 0; c < channels; c++) {
                    int src_idx = c * height * width + new_y * width + new_x;
                    int dst_idx = c * height * width + y * width + x;
                    rotated->data[dst_idx] = image->data[src_idx];
                }
            }
        }
    }
    
    return rotated;
}