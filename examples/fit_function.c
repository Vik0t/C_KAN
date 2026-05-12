//
// Created by viktor on 5/4/26.
//
#include "ckan.h"

int main() {
    Dataset* data = dataset_create_function_1d(sin_func, -1, 1, 100);


    int sizes[] = {1, 16, 1};
    Model *mlp = mlp_create(sizes, 3);
    Model* kan = kan_create(sizes, 3, 20);

    train(mlp, data, 1000, 0.01, 0);
    train(kan, data, 1000, 0.01, 0);
}