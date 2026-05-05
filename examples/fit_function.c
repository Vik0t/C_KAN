//
// Created by viktor on 5/4/26.
//
#include "ckan.h"

int main() {
    Dataset* data = dataset_create_function_1d(sin_func, -1, 1, 100);

    Model* mlp = mlp_create(1, 16, 1);
    Model* kan = kan_create(1, 20);

    train(mlp, data, 1000, 0.01);
    train(kan, data, 1000, 0.01);
}