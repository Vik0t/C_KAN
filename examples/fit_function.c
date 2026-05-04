//
// Created by viktor on 5/4/26.
//
#include "ckan.h"

int main() {
    Dataset* data = create_function_dataset(sin_func, 100);

    Model* mlp = mlp_create(1, 16, 1);
    Model* kan = kan_create(1, 20);

    train(mlp, data, 1000, 0.01);
    train(kan, data, 1000, 0.01);
}