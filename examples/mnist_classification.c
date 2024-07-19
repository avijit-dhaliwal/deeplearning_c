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

#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.01

#define TRAIN_IMAGES_URL "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
#define TRAIN_LABELS_URL "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
#define TEST_IMAGES_URL "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
#define TEST_LABELS_URL "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

void download_file(const char* url, const char* filename) {
    CURL *curl;
    FILE *fp;
    CURLcode res;

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(filename, "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
    }
}

void decompress_file(const char* compressed_filename, const char* decompressed_filename) {
    gzFile infile = gzopen(compressed_filename, "rb");
    FILE *outfile = fopen(decompressed_filename, "wb");
    char buffer[128];
    int num_read = 0;
    while ((num_read = gzread(infile, buffer, sizeof(buffer))) > 0) {
        fwrite(buffer, 1, num_read, outfile);
    }
    gzclose(infile);
    fclose(outfile);
}

Tensor* read_mnist_images(const char* filename) {
    FILE *file = fopen(filename, "rb");
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
    FILE *file = fopen(filename, "rb");
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
    download_file(TRAIN_IMAGES_URL, "train-images-idx3-ubyte.gz");
    download_file(TRAIN_LABELS_URL, "train-labels-idx1-ubyte.gz");
    download_file(TEST_IMAGES_URL, "t10k-images-idx3-ubyte.gz");
    download_file(TEST_LABELS_URL, "t10k-labels-idx1-ubyte.gz");

    decompress_file("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte");
    decompress_file("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte");
    decompress_file("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte");
    decompress_file("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte");

    // Read MNIST data
    *train_images = read_mnist_images("train-images-idx3-ubyte");
    *train_labels = read_mnist_labels("train-labels-idx1-ubyte");
    *test_images = read_mnist_images("t10k-images-idx3-ubyte");
    *test_labels = read_mnist_labels("t10k-labels-idx1-ubyte");
}

int main() {
    // Load MNIST dataset
    Tensor *train_images, *train_labels, *test_images, *test_labels;
    load_mnist(&train_images, &train_labels, &test_images, &test_labels);
    
    Dataset* train_dataset = dataset_create(train_images, train_labels);
    Dataset* test_dataset = dataset_create(test_images, test_labels);
    
    DataLoader* train_loader = dataloader_create(train_dataset, BATCH_SIZE, true);
    DataLoader* test_loader = dataloader_create(test_dataset, BATCH_SIZE, false);
    
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
    Optimizer* optimizer = optim_sgd(model, LEARNING_RATE);
    Loss* criterion = loss_cross_entropy();
    
    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
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