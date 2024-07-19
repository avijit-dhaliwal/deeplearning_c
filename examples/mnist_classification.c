#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include "../include/loss.h"
#include "../include/data.h"
#include "../include/train.h"
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

typedef struct {
    char* data;
    size_t size;
} DownloadedData;

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

Tensor* read_mnist_images(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    int magic_number, num_images, num_rows, num_cols;
    fread(&magic_number, sizeof(int), 1, file);
    fread(&num_images, sizeof(int), 1, file);
    fread(&num_rows, sizeof(int), 1, file);
    fread(&num_cols, sizeof(int), 1, file);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    size_t shape[] = {num_images, num_rows * num_cols};
    Tensor* images = tensor_create(NULL, shape, 2, (Device){CPU, 0});

    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < num_rows * num_cols; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, file);
            images->data[i * num_rows * num_cols + j] = pixel / 255.0f;
        }
    }

    fclose(file);
    return images;
}

Tensor* read_mnist_labels(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    int magic_number, num_items;
    fread(&magic_number, sizeof(int), 1, file);
    fread(&num_items, sizeof(int), 1, file);

    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    size_t shape[] = {num_items, 10};  // One-hot encoded labels
    Tensor* labels = tensor_create(NULL, shape, 2, (Device){CPU, 0});
    tensor_fill_(labels, 0.0f);

    for (int i = 0; i < num_items; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        labels->data[i * 10 + label] = 1.0f;
    }

    fclose(file);
    return labels;
}

void load_mnist(Tensor** train_images, Tensor** train_labels, Tensor** test_images, Tensor** test_labels) {
    // Download and decompress files
    DownloadedData train_images_data = download_file(MNIST_TRAIN_IMAGES_URL);
    DownloadedData train_labels_data = download_file(MNIST_TRAIN_LABELS_URL);
    DownloadedData test_images_data = download_file(MNIST_TEST_IMAGES_URL);
    DownloadedData test_labels_data = download_file(MNIST_TEST_LABELS_URL);

    size_t train_images_size, train_labels_size, test_images_size, test_labels_size;
    char* train_images_decompressed = decompress_gzip(train_images_data.data, train_images_data.size, &train_images_size);
    char* train_labels_decompressed = decompress_gzip(train_labels_data.data, train_labels_data.size, &train_labels_size);
    char* test_images_decompressed = decompress_gzip(test_images_data.data, test_images_data.size, &test_images_size);
    char* test_labels_decompressed = decompress_gzip(test_labels_data.data, test_labels_data.size, &test_labels_size);

    // Save decompressed data to files
    FILE* file;
    file = fopen("train-images-idx3-ubyte", "wb");
    fwrite(train_images_decompressed, 1, train_images_size, file);
    fclose(file);

    file = fopen("train-labels-idx1-ubyte", "wb");
    fwrite(train_labels_decompressed, 1, train_labels_size, file);
    fclose(file);

    file = fopen("t10k-images-idx3-ubyte", "wb");
    fwrite(test_images_decompressed, 1, test_images_size, file);
    fclose(file);

    file = fopen("t10k-labels-idx1-ubyte", "wb");
    fwrite(test_labels_decompressed, 1, test_labels_size, file);
    fclose(file);

    // Read MNIST data
    *train_images = read_mnist_images("train-images-idx3-ubyte");
    *train_labels = read_mnist_labels("train-labels-idx1-ubyte");
    *test_images = read_mnist_images("t10k-images-idx3-ubyte");
    *test_labels = read_mnist_labels("t10k-labels-idx1-ubyte");

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

int main() {
    // Load MNIST dataset
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
    
    return 0;
}