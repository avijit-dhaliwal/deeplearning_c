#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include "../include/autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>

#define MNIST_TRAIN_IMAGES_URL "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
#define MNIST_TRAIN_LABELS_URL "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
#define MNIST_TEST_IMAGES_URL "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
#define MNIST_TEST_LABELS_URL "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000
#define MNIST_IMAGE_SIZE 784
#define MNIST_LABEL_COUNT 10

// Struct to hold downloaded data
typedef struct {
    char* data;
    size_t size;
} DownloadedData;

// Callback function for libcurl
size_t write_data(void* ptr, size_t size, size_t nmemb, DownloadedData* data) {
    size_t new_size = data->size + size * nmemb;
    data->data = realloc(data->data, new_size + 1);
    if (data->data == NULL) {
        fprintf(stderr, "realloc() failed\n");
        exit(EXIT_FAILURE);
    }
    memcpy(data->data + data->size, ptr, size * nmemb);
    data->data[new_size] = '\0';
    data->size = new_size;

    return size * nmemb;
}

// Function to download a file
DownloadedData download_file(const char* url) {
    CURL* curl;
    CURLcode res;
    DownloadedData data = {0};

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            exit(EXIT_FAILURE);
        }
        curl_easy_cleanup(curl);
    }

    return data;
}

// Function to decompress gzip data
char* decompress_gzip(const char* compressed_data, size_t compressed_size, size_t* decompressed_size) {
    z_stream strm = {0};
    strm.next_in = (Bytef*)compressed_data;
    strm.avail_in = compressed_size;

    inflateInit2(&strm, 16 + MAX_WBITS);

    size_t buffer_size = compressed_size * 2;
    char* buffer = malloc(buffer_size);
    strm.next_out = (Bytef*)buffer;
    strm.avail_out = buffer_size;

    int ret;
    do {
        ret = inflate(&strm, Z_NO_FLUSH);
        if (ret == Z_STREAM_ERROR) {
            inflateEnd(&strm);
            free(buffer);
            return NULL;
        }
        if (strm.avail_out == 0) {
            buffer_size *= 2;
            buffer = realloc(buffer, buffer_size);
            strm.next_out = (Bytef*)(buffer + strm.total_out);
            strm.avail_out = buffer_size - strm.total_out;
        }
    } while (ret != Z_STREAM_END);

    inflateEnd(&strm);
    *decompressed_size = strm.total_out;
    return buffer;
}

// Function to load MNIST data
void load_mnist(Tensor** train_images, Tensor** train_labels, Tensor** test_images, Tensor** test_labels) {
    DownloadedData train_images_data = download_file(MNIST_TRAIN_IMAGES_URL);
    DownloadedData train_labels_data = download_file(MNIST_TRAIN_LABELS_URL);
    DownloadedData test_images_data = download_file(MNIST_TEST_IMAGES_URL);
    DownloadedData test_labels_data = download_file(MNIST_TEST_LABELS_URL);

    size_t train_images_size, train_labels_size, test_images_size, test_labels_size;
    char* train_images_decompressed = decompress_gzip(train_images_data.data, train_images_data.size, &train_images_size);
    char* train_labels_decompressed = decompress_gzip(train_labels_data.data, train_labels_data.size, &train_labels_size);
    char* test_images_decompressed = decompress_gzip(test_images_data.data, test_images_data.size, &test_images_size);
    char* test_labels_decompressed = decompress_gzip(test_labels_data.data, test_labels_data.size, &test_labels_size);

    // Parse train images
    *train_images = tensor_create(NULL, (size_t[]){MNIST_TRAIN_SIZE, MNIST_IMAGE_SIZE}, 2);
    memcpy((*train_images)->data, train_images_decompressed + 16, MNIST_TRAIN_SIZE * MNIST_IMAGE_SIZE * sizeof(float));

    // Parse train labels
    *train_labels = tensor_create(NULL, (size_t[]){MNIST_TRAIN_SIZE}, 1);
    memcpy((*train_labels)->data, train_labels_decompressed + 8, MNIST_TRAIN_SIZE * sizeof(float));

    // Parse test images
    *test_images = tensor_create(NULL, (size_t[]){MNIST_TEST_SIZE, MNIST_IMAGE_SIZE}, 2);
    memcpy((*test_images)->data, test_images_decompressed + 16, MNIST_TEST_SIZE * MNIST_IMAGE_SIZE * sizeof(float));

    // Parse test labels
    *test_labels = tensor_create(NULL, (size_t[]){MNIST_TEST_SIZE}, 1);
    memcpy((*test_labels)->data, test_labels_decompressed + 8, MNIST_TEST_SIZE * sizeof(float));

    // Normalize image data
    for (size_t i = 0; i < MNIST_TRAIN_SIZE * MNIST_IMAGE_SIZE; i++) {
        (*train_images)->data[i] /= 255.0f;
    }
    for (size_t i = 0; i < MNIST_TEST_SIZE * MNIST_IMAGE_SIZE; i++) {
        (*test_images)->data[i] /= 255.0f;
    }

    // Free downloaded and decompressed data
    free(train_images_data.data);
    free(train_labels_data.data);
    free(test_images_data.data);
    free(test_labels_data.data);
    free(train_images_decompressed);
    free(train_labels_decompressed);
    free(test_images_decompressed);
    free(test_labels_decompressed);
}

// Function to create the model
Sequential* create_model() {
    Layer** layers = malloc(3 * sizeof(Layer*));
    layers[0] = nn_linear(MNIST_IMAGE_SIZE, 128);
    layers[1] = nn_linear(128, 64);
    layers[2] = nn_linear(64, MNIST_LABEL_COUNT);
    return nn_sequential(layers, 3);
}

// Function to train the model
void train(Sequential* model, Tensor* train_images, Tensor* train_labels, size_t epochs, size_t batch_size) {
    Tensor** params = malloc(3 * sizeof(Tensor*));
    params[0] = model->layers[0]->weights;
    params[1] = model->layers[1]->weights;
    params[2] = model->layers[2]->weights;
    
    Optimizer* optimizer = optim_adam(params, 3, 0.001, 0.9, 0.999, 1e-8);
    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        
        for (size_t i = 0; i < MNIST_TRAIN_SIZE; i += batch_size) {
            size_t current_batch_size = (i + batch_size > MNIST_TRAIN_SIZE) ? (MNIST_TRAIN_SIZE - i) : batch_size;
            
            Tensor* batch_images = tensor_slice(train_images, i, current_batch_size);
            Tensor* batch_labels = tensor_slice(train_labels, i, current_batch_size);
            
            Node* input = autograd_variable(batch_images);
            Node* target = autograd_variable(batch_labels);
            
            // Forward pass
            Node* hidden1 = autograd_relu(autograd_matmul(input, autograd_variable(model->layers[0]->weights)));
            Node* hidden2 = autograd_relu(autograd_matmul(hidden1, autograd_variable(model->layers[1]->weights)));
            Node* output = autograd_softmax(autograd_matmul(hidden2, autograd_variable(model->layers[2]->weights)));
            
            // Compute loss
            Node* loss = autograd_cross_entropy(output, target);
            total_loss += tensor_sum(loss->data);
            
            // Backward pass
            autograd_backward(loss);
            
            // Update weights
            optimizer_step(optimizer);
            optimizer_zero_grad(optimizer);
            
            // Free memory
            tensor_free(batch_images);
            tensor_free(batch_labels);
            // Free autograd nodes (not implemented yet)
        }
        
        printf("Epoch %zu, Loss: %f\n", epoch + 1, total_loss / MNIST_TRAIN_SIZE);
    }
    
    free(params);
    optimizer_free(optimizer);
}

// Function to evaluate the model
float evaluate(Sequential* model, Tensor* test_images, Tensor* test_labels) {
    size_t correct = 0;
    
    for (size_t i = 0; i < MNIST_TEST_SIZE; i++) {
        Tensor* image = tensor_slice(test_images, i, 1);
        Tensor* label = tensor_slice(test_labels, i, 1);
        
        Tensor* output = sequential_forward(model, image);
        size_t predicted = tensor_argmax(output);
        
        if (predicted == (size_t)label->data[0]) {
            correct++;
        }
        
        tensor_free(image);
        tensor_free(label);
        tensor_free(output);
    }
    
    return (float)correct / MNIST_TEST_SIZE;
}

int main() {
    Tensor *train_images, *train_labels, *test_images, *test_labels;
    load_mnist(&train_images, &train_labels, &test_images, &test_labels);
    
    Sequential* model = create_model();
    
    train(model, train_images, train_labels, 10, 64);
    
    float accuracy = evaluate(model, test_images, test