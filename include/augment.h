#ifndef AUGMENT_H
#define AUGMENT_H

#include "tensor.h"

Tensor* augment_random_crop(Tensor* image, int crop_size);
Tensor* augment_random_flip(Tensor* image);
Tensor* augment_random_rotation(Tensor* image, float max_angle);

#endif // AUGMENT_H