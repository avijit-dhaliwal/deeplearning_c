# DeepLearning-C

A comprehensive deep learning framework implemented in C, inspired by PyTorch. This library provides a flexible and efficient foundation for building and training neural networks, with support for both CPU and GPU computations.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Library Overview](#library-overview)
   - [Tensor Operations](#tensor-operations)
   - [Neural Network Layers](#neural-network-layers)
   - [Optimizers](#optimizers)
   - [Loss Functions](#loss-functions)
   - [Data Handling](#data-handling)
   - [Training Utilities](#training-utilities)
5. [Usage Examples](#usage-examples)
   - [Basic Tensor Operations](#basic-tensor-operations)
   - [Creating a Neural Network](#creating-a-neural-network)
   - [Training a Model](#training-a-model)
   - [Saving and Loading Models](#saving-and-loading-models)
6. [Advanced Use Cases](#advanced-use-cases)
   - [Image Classification](#image-classification)
   - [Natural Language Processing](#natural-language-processing)
   - [Reinforcement Learning](#reinforcement-learning)
7. [Performance Optimization](#performance-optimization)
8. [Contributing](#contributing)
9. [License](#license)
10. [Insights from Implementing PyTorch in C](#insights-from-implementing-pytorch-in-c)

## Features

- Tensor operations with CPU and CUDA support
- Neural network layers:
  - Linear (Fully Connected)
  - Convolutional (Conv2D)
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Pooling (MaxPool2D)
  - Normalization (BatchNorm2D)
  - Dropout
- Optimizers:
  - Stochastic Gradient Descent (SGD)
  - Adam
- Loss functions:
  - Mean Squared Error (MSE)
  - Cross-Entropy
- Data loading and preprocessing utilities
- Training and evaluation utilities
- Model serialization for saving and loading

## Prerequisites

- GCC compiler (version 5.0 or later)
- Make build system
- CUDA Toolkit (optional, version 10.0 or later for GPU support)
- libcurl (for downloading datasets)
- zlib (for decompressing datasets)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/avijit-dhaliwal/deeplearning_c.git
cd deeplearning_c
```
2. Build the library, examples, and tests:
```bash
make
```
3. (Optional) To build with CUDA support, edit the Makefile and uncomment the CUDA-related lines, then run:
```bash
make clean
make
```

4. This is currently the only way to use this library. I am working on making this available with `pip` and `brew`.

## Library Overview

### Tensor Operations

The `Tensor` struct is the core data structure of the library, representing multi-dimensional arrays. Key operations include:

- Creation: `tensor_create()`
- Element-wise operations: `tensor_add()`, `tensor_sub()`, `tensor_mul()`, `tensor_div()`
- Matrix multiplication: `tensor_matmul()`
- Reshaping and transposition: `tensor_reshape()`, `tensor_transpose()`
- Device transfer: `tensor_to()`

Example:
```c
size_t shape[] = {2, 3};
Tensor* a = tensor_create(NULL, shape, 2, (Device){CPU, 0});
tensor_fill_(a, 1.0f);
Tensor* b = tensor_create(NULL, shape, 2, (Device){CPU, 0});
tensor_fill_(b, 2.0f);
Tensor* c = tensor_add(a, b);
```

### Neural Network Layers
The library provides various layer types to construct neural networks:
```c
nn_linear(): Fully connected layer
nn_conv2d(): 2D convolutional layer
nn_relu(), nn_sigmoid(), nn_tanh(): Activation functions
nn_maxpool2d(): 2D max pooling layer
nn_batchnorm2d(): 2D batch normalization layer
nn_dropout(): Dropout layer
```
Example:
```c
Module* linear = nn_linear(784, 128);
Module* relu = nn_relu();
Module* dropout = nn_dropout(0.5);
```
### Optimizers
Optimizers update the model parameters during training:
```c
optim_sgd(): Stochastic Gradient Descent
optim_adam(): Adam optimizer
```
Example:
```c
Optimizer* optimizer = optim_adam(model, 0.001, 0.9, 0.999, 1e-8);
```
### Loss Functions
Loss functions measure the difference between predictions and targets:
```c
loss_mse(): Mean Squared Error
loss_cross_entropy(): Cross-Entropy Loss
```
Example:
```c
Loss* criterion = loss_cross_entropy();
```
### Data Handling

The library provides utilities for data management:

- Dataset: Holds features and labels
- DataLoader: Iterates over datasets, providing batches for training

Example:
```c
Dataset* train_dataset = dataset_create(train_features, train_labels);
DataLoader* train_loader = dataloader_create(train_dataset, 64, true);
```
### Training Utilities
Functions to simplify the training process:

`train_epoch()`: Performs one epoch of training

`evaluate()`: Evaluates model performance on a dataset

## Usage Examples

Basic Tensor Operations

```c
// Create tensors
size_t shape[] = {2, 3};
Tensor* a = tensor_create(NULL, shape, 2, (Device){CPU, 0});
tensor_fill_(a, 1.0f);
Tensor* b = tensor_create(NULL, shape, 2, (Device){CPU, 0});
tensor_fill_(b, 2.0f);

// Perform operations
Tensor* c = tensor_add(a, b);
Tensor* d = tensor_matmul(a, tensor_transpose(b));

// Print results
tensor_print(c);
tensor_print(d);

// Free memory
tensor_free(a);
tensor_free(b);
tensor_free(c);
tensor_free(d);
Creating a Neural Network
// Define layers
Module* layers[] = {
    nn_linear(784, 128),
    nn_relu(),
    nn_linear(128, 64),
    nn_relu(),
    nn_linear(64, 10)
};

// Create sequential model
Sequential* model = nn_sequential(layers, 5);

// Create input tensor
size_t input_shape[] = {1, 784};
Tensor* input = tensor_create(NULL, input_shape, 2, (Device){CPU, 0});
tensor_fill_(input, 1.0f);

// Forward pass
Tensor* output = sequential_forward(model, input);

// Print output
tensor_print(output);

// Free memory
sequential_free(model);
tensor_free(input);
tensor_free(output);
Training a Model
// Load data
Tensor *train_images, *train_labels, *test_images, *test_labels;
load_mnist(&train_images, &train_labels, &test_images, &test_labels);

Dataset* train_dataset = dataset_create(train_images, train_labels);
Dataset* test_dataset = dataset_create(test_images, test_labels);

DataLoader* train_loader = dataloader_create(train_dataset, 64, true);
DataLoader* test_loader = dataloader_create(test_dataset, 64, false);

// Create model
Module* layers[] = {
    nn_linear(784, 128),
    nn_relu(),
    nn_linear(128, 64),
    nn_relu(),
    nn_linear(64, 10)
};
Sequential* model = nn_sequential(layers, 5);

// Create optimizer and loss function
Optimizer* optimizer = optim_adam(model, 0.001, 0.9, 0.999, 1e-8);
Loss* criterion = loss_cross_entropy();

// Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    TrainResult result = train_epoch(model, optimizer, criterion, train_loader, test_loader);
    printf("Epoch %d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f\n",
           epoch + 1, result.train_loss, result.train_accuracy, result.val_loss, result.val_accuracy);
}

// Clean up
sequential_free(model);
optimizer_free(optimizer);
loss_free(criterion);
dataloader_free(train_loader);
dataloader_free(test_loader);
dataset_free(train_dataset);
dataset_free(test_dataset);
tensor_free(train_images);
tensor_free(train_labels);
tensor_free(test_images);
tensor_free(test_labels);
Saving and Loading Models
// Save model
save_model(model, "mnist_model.bin");

// Load model
Sequential* loaded_model = load_model("mnist_model.bin");
```
## Advanced Use Cases

Image Classification

The library can be used for various image classification tasks, such as:

- MNIST handwritten digit recognition (as shown in the example)
- CIFAR-10 object classification
- ImageNet large-scale image classification

To adapt the MNIST example for other datasets, modify the data loading function and adjust the model architecture as needed.

### Natural Language Processing
For NLP tasks, you can implement:

- Text classification
- Sentiment analysis
- Language modeling

Example architecture for text classification:
```c
Module* layers[] = {
    nn_embedding(vocab_size, embedding_dim),
    nn_lstm(embedding_dim, hidden_dim),
    nn_linear(hidden_dim, num_classes)
};
Sequential* model = nn_sequential(layers, 3);
```
### Reinforcement Learning

The library can be used to implement neural networks for reinforcement learning algorithms, such as:

- Deep Q-Networks (DQN)
- Policy Gradient methods
- Actor-Critic algorithms

### Example DQN architecture:

```c
Module* layers[] = {
    nn_linear(state_dim, 64),
    nn_relu(),
    nn_linear(64, 64),
    nn_relu(),
    nn_linear(64, action_dim)
};
Sequential* q_network = nn_sequential(layers, 5);
```
## Performance Optimization
To optimize performance:

- Use CUDA support for GPU acceleration
- Experiment with different batch sizes
- Utilize model parallelism for large networks
- Implement data parallelism for distributed training
- Profile your code to identify bottlenecks

## Insights from Implementing PyTorch in C

For a detailed discussion on the insights gained from implementing this PyTorch-like library in C, check out my blog post: [Lessons Learned from Implementing PyTorch in C](https://avijitdhaliwal.com/post.html?id=deeplearning-c)

Key takeaways include:

- Understanding the underlying tensor operations
- Insights into automatic differentiation
- Challenges in memory management
- Considerations for GPU acceleration



## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

Any questions or feedback are welcome and appreciated. 

avijit.dhaliwal@gmail.com

## License
This project is licensed under the MIT License - see the LICENSE file for details.

